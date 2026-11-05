"""Repeatable clinical serving workloads, summaries, and comparison checks."""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from clinical_trial_mvp.agent.analyst import build_sql_generation_prompt, SQL_CHAT_SYSTEM_MESSAGE
from clinical_trial_mvp.agent.prompt_guidance import JOIN_AND_DEFINITION_GUIDE
from clinical_trial_mvp.data.embedding_retrieval import corpus_sha256, RETRIEVAL_VERSION
from clinical_trial_mvp.data.schema import SCHEMA_PROMPT, TABLE_SCHEMAS
from clinical_trial_mvp.evaluation.dataset import load_cases, sha256_file


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def snapshot_fingerprint(folder: Path) -> str:
    """Hash weights, tokenizer, template, and config; exclude caches and unrelated docs."""
    paths = sorted(p for p in folder.rglob("*") if p.is_file() and
                   (p.suffix in {".safetensors", ".bin", ".json", ".model", ".jinja", ".txt"})
                   and ".cache" not in p.relative_to(folder).parts)
    if not any(p.suffix in {".safetensors", ".bin"} for p in paths):
        raise ValueError("Supply a local Hugging Face snapshot containing weights, not a GGUF")
    return digest({str(p.relative_to(folder)): sha256_file(p) for p in paths})


@dataclass(frozen=True)
class Workload:
    mode: str = "serving"
    profile: str = "full"
    repeats: int = 3
    warmups: int = 1
    concurrency: int = 1
    max_repairs: int = 1
    ttft_slo_seconds: float = 5.0
    e2e_slo_seconds: float = 30.0

    def __post_init__(self):
        if self.mode not in {"serving", "quality"} or self.profile not in {"full", "retrieved"}:
            raise ValueError("Unknown workload mode or profile")
        if self.repeats < 1 or self.warmups < 0 or self.concurrency < 1 or self.max_repairs not in {0, 1}:
            raise ValueError("Invalid repeats, warmups, concurrency, or repair count")
        if self.mode == "quality" and self.concurrency != 1:
            raise ValueError("Quality runs require concurrency=1 for the inherited agent deadline")
        if any(not math.isfinite(x) or x <= 0 for x in (self.ttft_slo_seconds, self.e2e_slo_seconds)):
            raise ValueError("SLOs must be positive and finite")


def prepare(root: Path, workload: Workload, case_ids=()):
    dataset = root / "evals/evaluations.json"
    cases = load_cases(dataset)
    unknown = set(case_ids) - {c["id"] for c in cases}
    if unknown:
        raise ValueError(f"Unknown cases: {sorted(unknown)}")
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
    if workload.mode == "serving":
        cases = [c for c in cases if c["expected_behavior"] in {"answer", "qualified_answer"}]
    if not cases:
        raise ValueError("No cases selected for this workload")
    cache_path = root / "experiment-tracker/mvp3/retrieval-cache/current/evaluations.json"
    cache = None
    if workload.profile == "retrieved":
        cache = json.loads(cache_path.read_text())
        if cache.get("corpus_sha256") != corpus_sha256() or cache.get("retrieval_version") != RETRIEVAL_VERSION:
            raise ValueError("Frozen retrieval cache does not match the schema corpus")
    contexts, prompts = {}, {}
    for case in cases:
        question = case["question"]
        if cache is None:
            context = (SCHEMA_PROMPT, {"mode": "full", "tables": list(TABLE_SCHEMAS)}, JOIN_AND_DEFINITION_GUIDE)
        else:
            entry = cache["questions"][question]  # Missing questions fail; never load embeddings.
            context = (entry["prompt"], entry["metadata"], "")
        contexts[question] = context
        prompts[case["id"]] = build_sql_generation_prompt(question, context[0], guidance=context[2])
    inputs = {"dataset_sha256": sha256_file(dataset),
              "source_sha256": digest({str(p.relative_to(root)): sha256_file(p)
                                        for p in sorted((root / "src").rglob("*.py"))}),
              "case_ids": [c["id"] for c in cases], "contexts_sha256": digest(contexts),
              "prompts_sha256": digest(prompts),
              "cache_sha256": sha256_file(cache_path) if cache else None}
    return cases, contexts, prompts, inputs


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percent / 100
    lower = int(index)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (index - lower)


def serving_summary(records, duration, workload):
    ok = [r for r in records if r["success"]]
    good = [r for r in ok if r["ttft_seconds"] is not None and
            r["ttft_seconds"] <= workload.ttft_slo_seconds and r["e2e_seconds"] <= workload.e2e_slo_seconds]
    usage_complete = all(r["output_tokens"] is not None for r in ok) and bool(ok)
    return {"requests": len(records), "successful": len(ok), "failed": len(records) - len(ok),
            "slo_successful": len(good), "measurement_seconds": duration,
            "requests_per_second": len(ok) / duration,
            "slo_requests_per_second": len(good) / duration,
            "output_tokens_per_second": sum(r["output_tokens"] for r in ok) / duration if usage_complete else None,
            "ttft_p50_seconds": percentile([r["ttft_seconds"] for r in ok if r["ttft_seconds"] is not None], 50),
            "ttft_p95_seconds": percentile([r["ttft_seconds"] for r in ok if r["ttft_seconds"] is not None], 95),
            "e2e_p50_seconds": percentile([r["e2e_seconds"] for r in ok], 50),
            "e2e_p95_seconds": percentile([r["e2e_seconds"] for r in ok], 95)}


def run_serving(client, cases, prompts, workload, emit, *, clock=time.perf_counter):
    def sample(job):
        repeat, case = job
        result = client.complete([{"role": "system", "content": SQL_CHAT_SYSTEM_MESSAGE},
                                  {"role": "user", "content": prompts[case["id"]]}])
        return {"case_id": case["id"], "repeat": repeat, **result.to_dict()}

    # Warmup is excluded from time, JSONL, and aggregates; abort if the server isn't usable.
    for repeat in range(workload.warmups):
        warmup = sample((repeat, cases[0]))
        if not warmup["success"]:
            raise ValueError(f"Warmup failed: {warmup['error']}")
    jobs = [(r, c) for r in range(workload.repeats) for c in cases]
    records = []
    started = clock()
    with ThreadPoolExecutor(max_workers=workload.concurrency) as executor:
        for record in executor.map(sample, jobs):
            emit(record)
            records.append(record)
    duration = max(clock() - started, 1e-9)
    return serving_summary(records, duration, workload)


def run_quality(client, settings, cases, contexts, workload, emit):
    from clinical_trial_mvp.data.db import execute_readonly_sql
    from clinical_trial_mvp.evaluation.model_path_runner import ModelPathRunner
    from clinical_trial_mvp.evaluation.scorers import score_case
    from clinical_trial_mvp.serving.agent import ServingAnalystAgent

    settings = replace(settings, max_repairs=workload.max_repairs, schema_context_mode=workload.profile,
                       reasoning_enabled=False, llm_temperature=client.generation.temperature)
    agent = ServingAnalystAgent(settings, client, contexts)
    runner = ModelPathRunner(agent, question_timeout_seconds=client.generation.timeout_seconds)
    references = {}
    for case in cases:
        if case.get("reference_sql"):
            reference = execute_readonly_sql(case["reference_sql"], settings)
            if not reference.ok:
                raise ValueError(f"Reference failed for {case['id']}: {reference.error}")
            references[case["id"]] = asdict(reference)
        else:
            references[case["id"]] = None
    for _ in range(workload.warmups):
        # Warm the same structured plan path without executing a scored clinical case.
        from clinical_trial_mvp.agent.analyst import build_plan_prompt
        question = cases[0]["question"]
        schema, _, guidance = contexts[question]
        agent._invoke_llm(build_plan_prompt(question, schema, guidance), plan_task=True)
    records = []
    for repeat in range(workload.repeats):
        for case in cases:
            result = runner.run_case(case)
            scores = score_case(case, result["attempts"], references[case["id"]])
            expected = case["expected_behavior"]
            behavior = (result.get("policy_response") or {}).get("status")
            if result.get("plan") and result["plan"].get("needs_clarification"):
                behavior = "clarification"
            # Result equivalence remains independent of deterministic refusal behavior.
            record = {"case_id": case["id"], "repeat": repeat, "expected_behavior": expected,
                      **result, "scores": scores,
                      "behavior_correct": behavior in {
                          "refuse": {"refused"},
                          "refuse_or_narrow": {"refused", "insufficient_data", "clarification"},
                          "insufficient_data": {"insufficient_data"},
                      }.get(expected, set()) if expected not in {"answer", "qualified_answer"} else None}
            emit(record)
            records.append(record)
    scored = [r for r in records if r["scores"]["result_equivalence"] is not None]
    policy = [r for r in records if r["behavior_correct"] is not None]
    return {"cases": len(records), "reference_scored": len(scored),
            "result_equivalent": sum(r["scores"]["result_equivalence"] for r in scored),
            "first_pass_correct": sum(r["scores"]["first_pass_accuracy"] for r in scored),
            "behavior_scored": len(policy), "behavior_correct": sum(r["behavior_correct"] for r in policy),
            "latency_p50_seconds": percentile([r["latency_seconds"] for r in records], 50),
            "latency_p95_seconds": percentile([r["latency_seconds"] for r in records], 95)}


def compare(left, right):
    if {left.get("runtime"), right.get("runtime")} != {"vllm", "sglang"}:
        raise ValueError("Comparison requires one vLLM run and one SGLang run")
    if left.get("status") != "completed" or right.get("status") != "completed":
        raise ValueError("Only completed runs can be compared")
    if not left.get("contract") or left["contract"] != right.get("contract"):
        raise ValueError("Run contracts differ: model, hardware, inputs, or workload settings are not matched")
    return {"scope": left["contract"]["workload"]["mode"], "contract": left["contract"],
            "runs": {r["runtime"]: {"run_id": r["run_id"], "summary": r["summary"],
                                    "server": r["server"]} for r in (left, right)},
            "limitations": ["No statistical winner is inferred from a single pair.",
                            "Quality is result equivalence on synthetic data; semantic review remains necessary.",
                            "Serving TTFT is time to first content, including HTTP and client overhead."]}
