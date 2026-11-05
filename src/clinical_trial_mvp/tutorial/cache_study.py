"""Serial, shared-prefix cold/warm requests, separate from clinical agent evaluation."""

from dataclasses import asdict, replace
import json
import time
import urllib.request

from clinical_trial_mvp.agent.analyst import (
    ClinicalTrialAnalystAgent, SQL_CHAT_SYSTEM_MESSAGE, build_sql_generation_prompt,
)
from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.data.db import execute_readonly_sql
from clinical_trial_mvp.data.schema import SCHEMA_PROMPT
from clinical_trial_mvp.evaluation.dataset import sha256_file
from clinical_trial_mvp.evaluation.scorers import score_case
from clinical_trial_mvp.serving.benchmark import Workload, digest, percentile, prepare, serving_summary
from clinical_trial_mvp.tutorial.memory import PeakMemory


def messages(question):
    # The existing prompt builder places rules, full schema and guidance BEFORE
    # the question. No per-question retrieval may alter that shared prefix.
    return [{"role": "system", "content": SQL_CHAT_SYSTEM_MESSAGE},
            {"role": "user", "content": build_sql_generation_prompt(question, SCHEMA_PROMPT)}]


def control(client, route, *, method="GET"):
    request = urllib.request.Request(client.base_url.removesuffix("/v1") + route, method=method)
    with client.opener(request, timeout=30) as response:
        if response.status != 200:
            raise ValueError(f"Server control {route} returned {response.status}")
        raw = response.read()
    return json.loads(raw) if raw else None


def ready(client):
    # Readiness is outside every request clock; launching/installing is external.
    control(client, "/health")


def reset_prefix(client):
    response = control(client, "/reset_prefix_cache", method="POST")
    if not isinstance(response, dict) or response.get("success") is not True:
        raise ValueError("Cold cache unverified: reset API must return success=true; "
                         "empty legacy acknowledgements cannot establish a cold request")
    return {"method": "POST /reset_prefix_cache", "success": True}


def sql_score(sample, case, reference, settings):
    sql, error, executed = None, sample.error, None
    if sample.success:
        try:
            sql = ClinicalTrialAnalystAgent._parse_sql_json(sample.content)["sql"]
            executed = execute_readonly_sql(sql, settings)
            if not executed.ok:
                error = executed.error
        except (ValueError, TypeError, KeyError) as exc:
            error = str(exc)
    attempt = {"sql": sql, "execution_success": bool(executed and executed.ok),
               "query_result": asdict(executed) if executed else None,
               "raw_model_response": sample.content, "error": error}
    return {"attempts": [attempt], "scores": score_case(case, [attempt], reference)}


def measure_requests(spec, client, *, root, gpu_uuid=None, reset=None,
                     memory_factory=None, clock=time.perf_counter):
    """Pair identical target requests with empty vs schema-primed KV cache.

    Each warm target is primed using a DIFFERENT case after another reset, so it
    cannot reuse its own complete prompt. SQL scoring runs after request timing.
    """
    if spec["concurrency"] != 1 or spec["profile"] != "full":
        raise ValueError("Cold/warm study requires serial requests and identical full schema")
    from clinical_trial_mvp.tutorial.harness import TUTORIAL_CASES
    workload = Workload(profile="full", repeats=spec["repeats"], warmups=0, concurrency=1)
    cases, _, _, inputs = prepare(root, workload, TUTORIAL_CASES)
    settings = replace(Settings.from_env(), project_root=root, db_path=root / "data/db/clinical_trials.duckdb")
    references = {}
    for case in cases:
        reference = execute_readonly_sql(case["reference_sql"], settings)
        if not reference.ok:
            raise ValueError("Reference SQL failed: " + case["id"])
        references[case["id"]] = asdict(reference)
    reset = reset or (lambda: reset_prefix(client))
    memory_factory = memory_factory or (lambda: PeakMemory(gpu_uuid=gpu_uuid))
    # Warm kernels/connection, then reset cache before any measured cache-on call.
    warmup = client.complete(messages(cases[0]["question"]))
    if not warmup.success:
        raise ValueError("Excluded engine warmup failed: " + str(warmup.error))
    records, priming = [], []
    for repeat in range(spec["repeats"]):
        for index, case in enumerate(cases):
            phases = ("cold", "warm") if spec["prefix_cache"] else ("cache_off",)
            for phase in phases:
                reset_evidence = reset() if spec["prefix_cache"] else None
                if spec["prefix_cache"] and (not reset_evidence or reset_evidence.get("success") is not True):
                    raise ValueError("Reset did not confirm success")
                if phase == "warm":
                    other = cases[(index + 1) % len(cases)]
                    prime = client.complete(messages(other["question"]))
                    priming.append({"repeat": repeat, "target": case["id"], "prime_case": other["id"],
                                    **prime.to_dict()})
                    if not prime.success:
                        raise ValueError("Shared-schema priming failed: " + str(prime.error))
                with memory_factory() as memory:
                    started = clock()
                    sample = client.complete(messages(case["question"]))
                    duration = clock() - started
                # No reset, priming, memory process launch, DB or startup time is
                # included in the individual transport's TTFT/E2E or duration.
                record = {"case_id": case["id"], "repeat": repeat, "phase": phase,
                          "reset": reset_evidence, "request_seconds": duration,
                          "memory": memory.result(), **sample.to_dict()}
                record.update(sql_score(sample, case, references[case["id"]], settings))
                records.append(record)
    summaries = {}
    from clinical_trial_mvp.tutorial.harness import semantic_scores
    review_path = root / "tutorial/semantic_reviews.json"
    reviews = json.loads(review_path.read_text()).get("reviews", []) if review_path.exists() else []
    for phase in dict.fromkeys(r["phase"] for r in records):
        rows = [r for r in records if r["phase"] == phase]
        summary = serving_summary(rows, max(sum(r["request_seconds"] for r in rows), 1e-9), workload)
        peaks = [r["memory"]["peak_bytes"] for r in rows]
        tpot = [(r["e2e_seconds"] - r["ttft_seconds"]) / (r["output_tokens"] - 1)
                for r in rows if r["success"] and r["ttft_seconds"] is not None
                and r["output_tokens"] and r["output_tokens"] > 1]
        summary.update(sql_executed=sum(r["attempts"][0]["execution_success"] for r in rows),
                       result_equivalent=sum(r["scores"]["result_equivalence"] is True for r in rows),
                       reference_scored=len(rows), tpot_p50_seconds=percentile(tpot, 50),
                       peak_gpu_bytes=max(peaks) if peaks and all(p is not None for p in peaks) else None)
        summary["semantic"] = semantic_scores(rows, reviews)
        summaries[phase] = summary
    prefix = build_sql_generation_prompt("", SCHEMA_PROMPT).split("Question:\n")[0]
    return {"format": "mvp4-cache-phases-v2", "inputs": inputs,
            "shared_prefix_sha256": digest([SQL_CHAT_SYSTEM_MESSAGE, prefix]),
            "schema_sha256": digest(SCHEMA_PROMPT), "phases": summaries, "records": records,
            "priming_records": priming, "startup_in_request_timing": False,
            "database_sha256": sha256_file(settings.db_path)}


def phase_table(results):
    lines = ["| Variant | Phase | SQL match | Semantic | TTFT p50/p95 s | E2E p50/p95 s | TPOT p50 s | tok/s | Peak VRAM GiB |",
             "|---|---|---|---|---|---|---|---|---|"]
    def number(value):
        return "—" if value is None else f"{value:.3f}"
    for name, result in results.items():
        study = result.get("cache_study") or {}
        for phase, s in study.get("phases", {}).items():
            peak = s["peak_gpu_bytes"]
            semantic = s.get("semantic", {})
            review = (f'{semantic.get("correct", 0)}/{semantic["reviewed"]} reviewed'
                      if semantic.get("reviewed") else "unreviewed")
            lines.append(f'| {name} | {phase} | {s["result_equivalent"]}/{s["reference_scored"]} | '
                         f'{review} | '
                         f'{number(s["ttft_p50_seconds"])}/{number(s["ttft_p95_seconds"])} | '
                         f'{number(s["e2e_p50_seconds"])}/{number(s["e2e_p95_seconds"])} | '
                         f'{number(s["tpot_p50_seconds"])} | {number(s["output_tokens_per_second"])} | '
                         f'{number(peak / 1024**3 if peak is not None else None)} |')
        if not study.get("phases"):
            lines.append(f'| {name} | {result.get("status", "pending")} | — | — | — | — | — | — | — |')
    return "\n".join(lines)
