"""Common tutorial entry point: change configuration/adapters, not the clinical app."""

from __future__ import annotations

import hashlib
import json
import platform
import re
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.evaluation.dataset import sha256_file
from clinical_trial_mvp.serving.benchmark import Workload, prepare, run_quality, run_serving
from clinical_trial_mvp.serving.client import Generation
from clinical_trial_mvp.serving.portable import portable_text, portable_value
from clinical_trial_mvp.tutorial.adapters import serving_adapter
from clinical_trial_mvp.tutorial.memory import PeakMemory
from clinical_trial_mvp.tutorial.nim import nim_gate

ROOT = Path(__file__).resolve().parents[3]
TUTORIAL_CASES = ("eval_001", "eval_002", "eval_005", "eval_007")


def load_configurations(root=ROOT):
    return json.loads((Path(root) / "tutorial/runtimes.json").read_text())


def semantic_scores(records, reviews):
    """Match human judgements to exact SQL; execution match never implies semantic correctness."""
    index = {}
    for item in reviews:
        if not item.get("reviewer") or not item.get("reviewed_at") or type(item.get("correct")) is not bool:
            raise ValueError("Semantic reviews require reviewer, reviewed_at, and a boolean verdict")
        key = item["case_id"], item["sql_sha256"]
        if key in index:
            raise ValueError("Duplicate semantic review")
        index[key] = item["correct"]
    judgements = []
    for record in records:
        attempts = record.get("attempts", [])
        sql = (attempts[-1].get("sql") or "").strip() if attempts else ""
        key = record["case_id"], hashlib.sha256(sql.encode()).hexdigest()
        if sql and key in index:
            judgements.append(index[key])
    return {"correct": sum(judgements), "reviewed": len(judgements), "total": len(records)}


def validate_provenance(spec, server):
    for key in ("engine_version", "hardware", "launch_settings"):
        if not server.get(key):
            raise ValueError("Missing serving provenance field: " + key)
    fingerprint = server.get("model_sha256") or server.get("deployment_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("Model/deployment fingerprint must be a SHA-256 digest")
    if not server.get("model_sha256") and spec["runtime"] != "nim":
        raise ValueError("This runtime requires a weights/tokenizer fingerprint")
    if server.get("runtime") != spec["runtime"]:
        raise ValueError("Runtime provenance does not match the adapter")
    if server.get("model") and server["model"] != spec["model"]:
        raise ValueError("Served model alias does not match the configuration")
    actual = server["launch_settings"]
    for key in ("context", "dtype", "prefix_cache", "quantization"):
        if key in spec and actual.get(key) != spec[key]:
            raise ValueError("Running server differs from requested setting: " + key)
    return server


def serve_and_benchmark(spec, *, execute=False, root=ROOT, case_ids=TUTORIAL_CASES,
                        client=None, server=None, memory=None):
    """Attach to one prestarted server, run serial clinical quality then concurrent serving.

    Default is a no-network plan. Injected clients are test evidence, never hardware results.
    Installation/deployment belongs in scripts, not in each experiment cell.
    """
    root, spec = Path(root), dict(spec)
    injected_client = client is not None
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    workload = Workload(profile=spec.get("profile", "retrieved"), repeats=spec.get("repeats", 2),
                        concurrency=spec.get("concurrency", 1), warmups=spec.get("warmups", 1))
    generation = Generation(max_tokens=spec.get("max_tokens", 1024), seed=42,
                            timeout_seconds=spec.get("timeout_seconds", 180))
    cases, contexts, prompts, inputs = prepare(root, workload, case_ids)
    result = {"run_id": run_id, "experiment": spec["experiment"], "runtime": spec["runtime"],
              "hardware": spec["hardware"], "configuration": spec, "inputs": inputs,
              "status": "plan_only", "evidence": "none", "quality": None, "serving": None,
              "memory": {"peak_bytes": None, "scope": "unmeasured"}, "records": {}}
    if not execute:
        result["reason"] = "Planned only; enable this hardware group after starting its server."
        return result
    try:
        if spec["runtime"] == "nim":
            path = root / spec.get("nim_profile", "tutorial/nim_profile.example.json")
            profile = json.loads(path.read_text()) if path.exists() else {}
            allowed, reason = nim_gate(profile, model=spec["model"], hardware=spec["hardware"],
                                       enabled=True, system=platform.system())
            if not allowed:
                result.update(status="skipped", reason=reason)
                return result
        if client is None:
            if spec.get("host") == "mac" and platform.system() != "Darwin":
                raise ValueError("This configuration requires the Mac host")
            if spec.get("host") == "cuda" and platform.system() != "Linux":
                raise ValueError("This configuration requires the Linux GPU host")
        server = server or json.loads((root / spec["provenance"]).read_text())
        result["server"] = validate_provenance(spec, server)
        settings = replace(Settings.from_env(), project_root=root,
                           db_path=root / "data/db/clinical_trials.duckdb")
        result["database_sha256"] = sha256_file(settings.db_path)
        import duckdb
        with duckdb.connect() as connection:
            result["evaluation_date"] = str(connection.execute("SELECT CURRENT_DATE").fetchone()[0])
        client = client or serving_adapter(spec["runtime"], spec["base_url"], spec["model"], generation,
            context=spec.get("context", 8192), structured_outputs=spec.get("structured_outputs", False))
        quality_records, serving_records = [], []
        memory = memory or PeakMemory(gpu_uuid=server.get("gpu_uuid"), server_pid=server.get("server_pid"))
        with memory:
            result["quality"] = run_quality(client, settings, cases, contexts,
                replace(workload, mode="quality", concurrency=1, repeats=1, warmups=0), quality_records.append)
            result["serving"] = run_serving(client, cases, prompts, workload, serving_records.append)
        result["memory"] = memory.result()
        result["quality"]["sql_executed"] = sum(
            bool(r["attempts"] and r["attempts"][-1].get("execution_success")) for r in quality_records)
        review_path = root / "tutorial/semantic_reviews.json"
        reviews = json.loads(review_path.read_text()).get("reviews", []) if review_path.exists() else []
        result["quality"]["semantic"] = semantic_scores(quality_records, reviews)
        counts = [r["prompt_tokens"] for r in serving_records]
        result["serving"]["mean_prompt_tokens"] = mean(counts) if all(c is not None for c in counts) else None
        result["records"] = {"quality": quality_records, "serving": serving_records}
        result.update(status="completed", evidence="hardware" if server.get("evidence") == "hardware" and not injected_client else "test_or_unverified")
        if result["evidence"] != "hardware":
            result["reason"] = "Injected or unverified provenance; excluded from hardware conclusions."
        result["finding"] = "Inspect execution match, semantic review coverage, and latency together."
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        result.update(status="failed", reason=portable_text(exc))
    if result["status"] == "completed" and result["evidence"] == "hardware":
        output = root / "experiment-tracker/tutorial/runs" / (run_id + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(portable_value(result), indent=2, default=str) + "\n")
    return portable_value(result)


def _number(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _notebook_summary(result):
    return (result.get("status") == "completed" and result.get("evidence") == "notebook_output"
            and result.get("format") == "tutorial-notebook-summary-v1")


def _displayable(result):
    return result.get("status") == "completed" and (result.get("evidence") == "hardware" or _notebook_summary(result))


def _display_number(result, value, digits=2):
    text = _number(value, digits)
    return "≈" + text if value is not None and _notebook_summary(result) else text


def _import_note(results):
    return (["", "≈ Imported from displayed notebook output, at its displayed precision. "
             "Raw request/SQL records, model/database fingerprints, run dates and launch provenance "
             "were not embedded; pairing remains unverified."]
            if any(_notebook_summary(r) for r in results) else [])


def summary_markdown(results):
    lines = ["| Experiment | Hardware | Runtime | Quality | TTFT | tok/s | Peak memory |",
             "|---|---|---|---|---|---|---|"]
    def order(result):
        name = result["experiment"]
        group = next((i for i, prefix in enumerate(("MVP1/", "MVP2/", "MVP3/", "MVP4/", "Optimization/", "ExecuTorch/"))
                      if name.startswith(prefix)), 6)
        return group, 0 if result.get("runtime") == "vllm" else 1, name
    for r in sorted(results, key=order):
        measured = _displayable(r)
        q, s, m = (r.get("quality") or {}), (r.get("serving") or {}), r.get("memory", {})
        quality = f"{q.get('result_equivalent', 0)}/{q.get('reference_scored', 0)} match" if measured and q else "—"
        peak = _display_number(r, m.get("peak_bytes") / 1024**3) + " GiB" if measured and m.get("peak_bytes") is not None else "—"
        label = r["experiment"] + " · " + ("notebook summary" if _notebook_summary(r) else r["status"])
        lines.append(f"| {label} | {r['hardware']} | {r['runtime']} | {quality} | "
                     f"{_display_number(r, s.get('ttft_p50_seconds')) if measured else '—'} | "
                     f"{_display_number(r, s.get('output_tokens_per_second')) if measured else '—'} | {peak} |")
    return "\n".join(lines + ([] if results else ["| Pending | — | — | — | — | — | — | "]) + _import_note(results))


def measurements_markdown(results):
    lines = ["| Configuration/status | SQL | Result match | Semantic review | Prompt tokens | TTFT p50 s | Latency p50/p95 s | tok/s | Peak memory |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        q, s = r.get("quality") or {}, r.get("serving") or {}
        if not _displayable(r):
            lines.append(f"| {r['experiment']} · {r['status']} | — | — | — | — | — | — | — | — |")
            continue
        review = q.get("semantic", {})
        semantic = (f"{review.get('correct', 0)}/{review.get('reviewed', 0)} reviewed; {review.get('total', 0)} total"
                    if review.get('reviewed') else f"Unreviewed; {review.get('total', 0)} total")
        peak = r.get("memory", {}).get("peak_bytes")
        label = r["experiment"] + (" · notebook summary" if _notebook_summary(r) else "")
        memory_text = _display_number(r, peak / 1024**3) + " GiB" if peak is not None else "—"
        lines.append(f"| {label} | {q.get('sql_executed', 0)}/{q.get('cases', 0)} | "
            f"{q.get('result_equivalent', 0)}/{q.get('reference_scored', 0)} | {semantic} | "
            f"{_display_number(r, s.get('mean_prompt_tokens'), 0)} | {_display_number(r, s.get('ttft_p50_seconds'))} | "
            f"{_display_number(r, s.get('e2e_p50_seconds'))}/{_display_number(r, s.get('e2e_p95_seconds'))} | "
            f"{_display_number(r, s.get('output_tokens_per_second'))} | {memory_text} |")
    return "\n".join(lines + _import_note(results))


def findings(results, trade_off, decision):
    measured = [r for r in results if r["status"] == "completed" and r.get("evidence") == "hardware"]
    observation = f"{len(measured)} measured configurations; review the table before choosing a winner." if measured else (
        "No hardware measurements yet. " + " ".join(dict.fromkeys(r.get("reason", "") for r in results)))
    imported = sum(_notebook_summary(r) for r in results)
    if imported:
        observation = (f"{len(measured)} full hardware records and {imported} imported notebook summaries. "
                       "Summaries retain displayed precision; raw provenance is required for a controlled comparison.")
    return f"**Observation:** {observation}  \n**Trade-off:** {trade_off}  \n**Decision:** {decision}"


def pair_status(results):
    if any(_notebook_summary(r) for r in results):
        return "Pairing unverified: imported notebook summaries lack full model/database fingerprints, run dates and launch provenance."
    measured = [r for r in results if r["status"] == "completed" and r.get("evidence") == "hardware"]
    if len(measured) < 2:
        return "Pairing pending: measure both configurations before drawing a comparison."
    for key in ("database_sha256", "evaluation_date"):
        if len({r.get(key) for r in measured}) > 1:
            return "Not paired: " + key + " differs."
    identities = {r.get("server", {}).get("model_sha256") for r in measured}
    if None in identities or len(identities) != 1:
        return "Not paired: identical weight/tokenizer fingerprints have not been established."
    if len({r["inputs"]["prompts_sha256"] for r in measured}) > 1:
        return "Context study: prompts differ intentionally; runtime speed alone is not isolated."
    return "Same model/tokenizer fingerprint and prompt workload; inspect hardware and launch settings too."


class TutorialSession:
    def __init__(self, root=ROOT):
        self.root, self.results = Path(root), []

    def run(self, spec, *, execute=False):
        if not execute:
            for old in self.results:
                if old["experiment"] != spec["experiment"] or not _displayable(old):
                    continue
                if (_notebook_summary(old) or
                        old.get("database_sha256") == sha256_file(self.root / "data/db/clinical_trials.duckdb")):
                    # Replay the historical configuration, not an unmeasured plan on this host.
                    return old
        result = serve_and_benchmark(spec, execute=execute, root=self.root)
        # Rerunning a cell updates its summary entry rather than duplicating it.
        self.results = [r for r in self.results if r["experiment"] != result["experiment"]]
        self.results.append(result)
        return result

    def summary(self):
        return summary_markdown(self.results)

    def load_saved(self):
        """Load measured tutorial records from the same synthetic database, keeping timestamps/provenance."""
        for path in sorted((self.root / "experiment-tracker/tutorial/runs").glob("*.json")):
            result = json.loads(path.read_text())
            if (result.get("status") == "completed" and result.get("evidence") == "hardware"
                    and result.get("database_sha256") == sha256_file(self.root / "data/db/clinical_trials.duckdb")):
                self.results = [r for r in self.results if r["experiment"] != result["experiment"]]
                self.results.append(result)
        for path in sorted((self.root / "experiment-tracker/tutorial/edge").glob("*.json")):
            result = json.loads(path.read_text())
            if (result.get("status") == "completed" and result.get("evidence") == "hardware"
                    and result.get("runtime") == "executorch" and result.get("experiment") == "ExecuTorch/edge"):
                self.results = [r for r in self.results if r["experiment"] != result["experiment"]]
                self.results.append(result)

        for path in sorted((self.root / "experiment-tracker/tutorial/imported").glob("*.json")):
            result = json.loads(path.read_text())
            if not _notebook_summary(result):
                continue
            source = result.get("source", {})
            archive_name = Path(source.get("archive", ""))
            if archive_name.is_absolute() or ".." in archive_name.parts:
                raise ValueError("Imported source archive must be a repository-relative path")
            archive = self.root / archive_name
            if not archive.is_file() or sha256_file(archive) != source.get("archived_sha256"):
                raise ValueError("Imported notebook source archive checksum mismatch")
            # Full same-database run records supersede rendered summaries.
            if any(r["experiment"] == result["experiment"] and r.get("evidence") == "hardware" for r in self.results):
                continue
            self.results = [r for r in self.results if r["experiment"] != result["experiment"]]
            self.results.append(result)
