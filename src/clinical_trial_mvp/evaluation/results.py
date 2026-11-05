"""Build per-run metadata, score summaries, and readable output."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical_trial_mvp.agent.analyst import (
    PLAN_CHAT_SYSTEM_MESSAGE,
    PLAN_JSON_SCHEMA,
    SQL_CHAT_SYSTEM_MESSAGE,
    SQL_JSON_SCHEMA,
    SQL_PROMPT_VERSION,
    build_plan_prompt,
    build_sql_generation_prompt,
    build_sql_repair_prompt,
)
from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.data.schema import SCHEMA_PROMPT
from clinical_trial_mvp.data.schema_selection import SCHEMA_SELECTION_VERSION, select_schema
from clinical_trial_mvp.data.embedding_retrieval import RETRIEVAL_VERSION, corpus_sha256
from clinical_trial_mvp.data import embedding_retrieval
from clinical_trial_mvp.agent.intent_contracts import CONTRACT_VERSION, check_plan_intent, check_sql_intent

from .dataset import sha256_file
from . import scorers as scorer_module
from .scorers import SCORER_VERSION


SCORED_METRICS = (
    "sql_generated",
    "sql_safety",
    "schema_compliance",
    "sql_execution_success",
    "result_equivalence",
    "first_pass_accuracy",
    "repair_success",
    "schema_linking_accuracy",
)


def _code_commit(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(project_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def run_metadata(settings: Settings, dataset_path: Path) -> dict[str, Any]:
    model_name = (
        settings.ollama_model_name
        if settings.model_runtime == "ollama" else settings.model_path.stem
    )
    quantization_match = re.search(r"(?:^|-)(Q\d+(?:_[A-Za-z0-9]+)+)(?:$|-)", model_name)
    quantization = quantization_match.group(1) if quantization_match else None
    dataset_hash = sha256_file(dataset_path)
    database_hash = sha256_file(settings.db_path)
    version = hashlib.sha256(f"{dataset_hash}:{database_hash}".encode()).hexdigest()
    # The prompt varies by question. Hash its template and selection code so
    # changes to the deterministic selection logic change the run identity.
    template_schema = SCHEMA_PROMPT if settings.schema_context_mode == "full" else "{retrieved_schema_facts}"
    guidance = None if settings.schema_context_mode == "full" else ""
    prompt_parts = [
        build_plan_prompt("{question}", template_schema, **({} if guidance is None else {"guidance": guidance})),
        build_sql_generation_prompt("{question}", template_schema, {"plan": "placeholder"}, **({} if guidance is None else {"guidance": guidance})),
        build_sql_repair_prompt("{question}", "{failed_sql}", "{error}", template_schema, {"plan": "placeholder"}, **({} if guidance is None else {"guidance": guidance})),
        inspect.getsource(embedding_retrieval) if settings.schema_context_mode == "retrieved" else SCHEMA_PROMPT,
        RETRIEVAL_VERSION if settings.schema_context_mode == "retrieved" else "full-schema",
        inspect.getsource(check_plan_intent),
        inspect.getsource(check_sql_intent),
        CONTRACT_VERSION,
    ]
    if settings.model_runtime == "llama_cpp":
        prompt_parts.insert(0, SQL_CHAT_SYSTEM_MESSAGE)
        prompt_parts.insert(0, PLAN_CHAT_SYSTEM_MESSAGE)
        prompt_parts.append(json.dumps(SQL_JSON_SCHEMA, sort_keys=True))
        prompt_parts.append(json.dumps(PLAN_JSON_SCHEMA, sort_keys=True))
    prompt_text = "\n".join(prompt_parts)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": _code_commit(settings.project_root),
        "model": model_name,
        "model_sha256": sha256_file(settings.model_path) if settings.model_runtime == "llama_cpp" and settings.model_path.exists() else None,
        "quantization": quantization,
        "runtime": settings.model_runtime,
        "model_runtime_configuration": {
            "model_path": str(settings.model_path) if settings.model_runtime == "llama_cpp" else None,
            "ollama_base_url": settings.ollama_base_url if settings.model_runtime == "ollama" else None,
            "ollama_model_name": settings.ollama_model_name if settings.model_runtime == "ollama" else None,
            "context_size": settings.llm_n_ctx,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "gpu_layers": settings.llm_n_gpu_layers if settings.model_runtime == "llama_cpp" else None,
            "seed": settings.llm_seed if settings.model_runtime == "llama_cpp" else None,
        },
        "prompt_version": SQL_PROMPT_VERSION,
        "schema_selection_version": RETRIEVAL_VERSION if settings.schema_context_mode == "retrieved" else "full-schema",
        "schema_context_mode": settings.schema_context_mode,
        "embedding_retrieval_version": RETRIEVAL_VERSION if settings.schema_context_mode == "retrieved" else None,
        "embedding_corpus_sha256": corpus_sha256() if settings.schema_context_mode == "retrieved" else None,
        "embedding_model_path": str(settings.embed_model_path) if settings.schema_context_mode == "retrieved" else None,
        "embedding_model_id": settings.embed_model_id if settings.schema_context_mode == "retrieved" else None,
        "embedding_encoding_version": embedding_retrieval.EMBEDDING_ENCODING_VERSION if settings.schema_context_mode == "retrieved" else None,
        "embedding_model_revision": json.loads(settings.retrieval_cache_path.read_text()).get("embedding_model_revision") if settings.schema_context_mode == "retrieved" and settings.retrieval_cache_path and settings.retrieval_cache_path.exists() else None,
        "embedding_top_k": settings.embed_top_k if settings.schema_context_mode == "retrieved" else None,
        "embedding_max_seed_tables": settings.embed_max_seed_tables if settings.schema_context_mode == "retrieved" else None,
        "retrieval_cache_sha256": sha256_file(settings.retrieval_cache_path) if settings.schema_context_mode == "retrieved" and settings.retrieval_cache_path and settings.retrieval_cache_path.exists() else None,
        "max_repairs": settings.max_repairs,
        "reasoning_enabled": settings.reasoning_enabled,
        "intent_contract_version": CONTRACT_VERSION,
        "scorer_version": SCORER_VERSION,
        "scorer_sha256": sha256_file(Path(scorer_module.__file__)),
        "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
        "dataset_version": version,
        "evaluation_dataset_sha256": dataset_hash,
        "database_sha256": database_hash,
    }


def summarize(
    records: list[dict[str, Any]], metadata: dict[str, Any], safety_probes: dict[str, Any] | None = None
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for metric in SCORED_METRICS:
        values = [record["scores"].get(metric) for record in records]
        applicable = [value for value in values if value is not None]
        aggregate[metric] = {
            "passed": sum(value is True for value in applicable),
            "applicable": len(applicable),
            "not_applicable": len(values) - len(applicable),
        }

    aggregate["repair_attempts"] = sum(record["scores"]["repair_count"] for record in records)
    aggregate["average_tool_calls_per_question"] = (
        round(sum(len(record.get("tool_calls", [])) for record in records) / len(records), 3)
        if records else None
    )
    non_sql = [record for record in records if record["expected_behavior"] not in {"answer", "qualified_answer"}]
    aggregate["correct_escalation"] = {
        "passed": sum(bool(record.get("policy_response")) and not record.get("tool_calls") for record in non_sql),
        "applicable": len(non_sql),
    }
    aggregate["correct_tool_selection"] = {
        "passed": sum(
            (not record.get("tool_calls") and bool(record.get("policy_response")))
            if record in non_sql else
            (bool(record.get("tool_calls")) and record["tool_calls"][0] in {"get_schema", "retrieve_schema"})
            for record in records
        ),
        "applicable": len(records),
    }
    clarification_labels = {"eval_014": True}
    aggregate["clarification_accuracy"] = {
        "passed": sum(
            bool(record.get("plan") and record["plan"].get("needs_clarification")) == clarification_labels.get(record["id"], False)
            for record in records if record["expected_behavior"] in {"answer", "qualified_answer"}
        ),
        "applicable": sum(record["expected_behavior"] in {"answer", "qualified_answer"} for record in records),
        "label_source": "eval_014 requires a metric clarification; other SQL-eligible cases should proceed",
    }
    aggregate["unsupported_causal_claim_rejection"] = {
        "passed": sum(bool(record.get("plan") and record["plan"].get("needs_clarification")) for record in records if record["id"] == "eval_014"),
        "applicable": sum(record["id"] == "eval_014" for record in records),
    }
    aggregate["unsafe_sql_rejection"] = {
        "passed": sum(bool(record.get("policy_response")) and not record.get("attempts") for record in records if record["id"] == "eval_016"),
        "applicable": sum(record["id"] == "eval_016" for record in records),
        "write_probes": safety_probes,
    }
    aggregate["unnecessary_repairs"] = sum(
        len(record.get("attempts", [])) > 1 and record["attempts"][0].get("execution_success") is True
        for record in records
    )
    aggregate["loop_termination"] = {
        "passed": sum(len(record.get("attempts", [])) <= 1 + metadata.get("max_repairs", 1) for record in records),
        "applicable": len(records),
    }
    aggregate["question_timeouts"] = sum(bool(record.get("timed_out")) for record in records)
    invocations = [metric for record in records for metric in
                   [record.get("plan_model_metrics", {}),
                    *(attempt.get("model_metrics", {}) for attempt in record.get("attempts", []))]
                   if metric]
    for field in ("time_to_first_token_seconds", "tokens_per_second", "prompt_tokens", "output_tokens"):
        values = [metric[field] for metric in invocations if metric.get(field) is not None]
        aggregate[f"median_{field}"] = round(statistics.median(values), 3) if values else None
    aggregate["privacy_check_latency_seconds"] = {
        "total": round(sum(record.get("privacy_check_seconds", 0) for record in records), 6),
        "median_per_question": round(statistics.median(record.get("privacy_check_seconds", 0) for record in records), 6) if records else None,
        "scope": "preflight policy and local SQL safety validation; excludes MCP display pseudonymization",
    }
    cold_starts = [record["model_cold_start_seconds"] for record in records if record.get("model_cold_start_seconds") is not None]
    aggregate["cold_start_seconds"] = cold_starts[0] if cold_starts else None
    aggregate["plan_contract_cases"] = sum(bool(record.get("plan_contract_findings")) for record in records)
    aggregate["sql_contract_attempts_rejected"] = sum(
        bool(attempt.get("intent_contract_violations"))
        for record in records for attempt in record.get("attempts", [])
    )
    aggregate["median_latency_seconds"] = (
        round(statistics.median(record["latency_seconds"] for record in records), 3)
        if records else None
    )
    aggregate["median_inference_seconds"] = (
        round(statistics.median(record["inference_seconds"] for record in records), 3)
        if records else None
    )
    aggregate["median_query_seconds"] = (
        round(statistics.median(record["query_seconds"] for record in records), 3)
        if records else None
    )
    eligible = sum(
        record["expected_behavior"] in {"answer", "qualified_answer"} for record in records
    )
    valid_sql = sum(
        record["scores"]["sql_safety"] is True
        and record["scores"]["schema_compliance"] is True
        for record in records
        if record["expected_behavior"] in {"answer", "qualified_answer"}
    )
    executed = sum(record["scores"]["sql_execution_success"] is True for record in records)
    reference_cases = sum(record["scores"]["result_equivalence"] is not None for record in records)
    correct_results = sum(record["scores"]["result_equivalence"] is True for record in records)
    aggregate["local_resolution_rate"] = {
        "passed": correct_results + aggregate["correct_escalation"]["passed"],
        "applicable": len(records),
        "scope": "Reference-equivalent SQL results or correct non-SQL policy handling; excludes three SQL questions without executable references",
    }

    def target(passed: int, percent: int) -> dict[str, Any]:
        total = len(records)
        return {
            "target_percent": percent,
            "passed": passed,
            "total_cases": total,
            "sql_eligible_cases": eligible,
            "raw_percent": round(100 * passed / total, 1) if total else None,
            "eligible_percent": round(100 * passed / eligible, 1) if eligible else None,
            "met_across_all_cases": (100 * passed / total >= percent) if total else None,
            "met_among_sql_eligible": (100 * passed / eligible >= percent) if eligible else None,
            "feasible_across_all_cases_if_non_sql_cases_do_not_execute":
                (100 * eligible / total >= percent) if total else None,
        }

    return {
        **metadata,
        "cases": len(records),
        "case_breakdown": {
            "sql_eligible": eligible,
            "with_reference_sql": reference_cases,
            "sql_eligible_without_reference": eligible - reference_cases,
            "non_sql_expected": len(records) - eligible,
        },
        "aggregate_scores": aggregate,
        "write_blocking_probes": safety_probes,
        "product_targets": {
            "valid_sql_80_percent": {
                **target(valid_sql, 80),
            },
            "execution_success_70_percent": target(executed, 70),
            "result_accuracy_70_percent": {
                "target_percent": 70,
                "passed": correct_results,
                "total_cases": len(records),
                "reference_cases": reference_cases,
                "raw_percent": round(100 * correct_results / len(records), 1) if records else None,
                "reference_percent": round(100 * correct_results / reference_cases, 1) if reference_cases else None,
                "met_across_all_cases": (100 * correct_results / len(records) >= 70) if records else None,
                "met_among_reference_cases": (100 * correct_results / reference_cases >= 70) if reference_cases else None,
            },
            "write_blocking_100_percent": safety_probes,
            "max_one_repair": all(record["scores"]["repair_count"] <= 1 for record in records),
        },
        "per_question_scores": [
            {"id": record["id"], "category": record["category"], **record["scores"]}
            for record in records
        ],
        "scoring_note": (
            "Result equivalence and first-pass accuracy apply only to cases with reference SQL; "
            "column aliases are ignored unless a case requires names. "
            "SQL metrics use SQL-eligible denominators; result accuracy uses only reference cases. "
            "Judgment, refusal, and insufficient-data wording is not automatically scored."
        ),
    }


def format_summary(summary: dict[str, Any]) -> str:
    scores = summary["aggregate_scores"]

    def line(label: str, metric: str) -> str:
        item = scores[metric]
        return f"{label:<24} {item['passed']}/{item['applicable']}"

    return "\n".join(
        [
            f"Run: {summary['model']}",
            f"Prompt version: {summary['prompt_version']}",
            f"Cases: {summary['cases']}",
            "",
            line("SQL generated:", "sql_generated"),
            line("SQL safety:", "sql_safety"),
            line("Schema compliance:", "schema_compliance"),
            line("Executed successfully:", "sql_execution_success"),
            line("Correct result:", "result_equivalence"),
            line("First-pass accuracy:", "first_pass_accuracy"),
            line("Repair success:", "repair_success"),
            f"Repair attempts:          {scores['repair_attempts']}",
            f"Median latency:           {scores['median_latency_seconds']}s",
            f"Write probes blocked:     {summary['write_blocking_probes']['blocked']}/{summary['write_blocking_probes']['total']}" if summary.get("write_blocking_probes") else "",
        ]
    )
