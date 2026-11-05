"""Append-only manifests and a rebuildable catalog for evaluation artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HISTORY_VERSION = 1


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_file(source: Path, directory: Path, suffix: str) -> Path:
    """Store one content-addressed copy per distinct dataset or database."""
    digest = file_sha256(source)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}{suffix}"
    if target.exists():
        if file_sha256(target) != digest:
            raise ValueError(f"Snapshot has changed: {target}")
        return target
    temporary = directory / f".{digest}.{os.getpid()}.tmp"
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        if file_sha256(temporary) != digest:
            raise ValueError(f"Source changed while snapshotting: {source}")
        # Another process may have completed the same snapshot first.
        if not target.exists():
            os.link(temporary, target)
        elif file_sha256(target) != digest:
            raise ValueError(f"Snapshot has changed: {target}")
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _artifact(path: Path | None, root: Path) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path}, line {number}") from exc
    return records


def _summary_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    scores = summary.get("aggregate_scores", {})
    return {
        "cases": summary.get("cases"),
        "sql_eligible": summary.get("case_breakdown", {}).get("sql_eligible"),
        "reference_cases": summary.get("case_breakdown", {}).get("with_reference_sql"),
        "sql_generated": scores.get("sql_generated"),
        "sql_safety": scores.get("sql_safety"),
        "schema_compliance": scores.get("schema_compliance"),
        "execution": scores.get("sql_execution_success"),
        "correct_result": scores.get("result_equivalence"),
        "first_pass_accuracy": scores.get("first_pass_accuracy"),
        "repair_success": scores.get("repair_success"),
        "repair_attempts": scores.get("repair_attempts"),
        "median_latency_seconds": scores.get("median_latency_seconds"),
        "question_timeouts": scores.get("question_timeouts"),
        "plan_contract_cases": scores.get("plan_contract_cases"),
        "sql_contract_attempts_rejected": scores.get("sql_contract_attempts_rejected"),
        "write_probes": summary.get("write_blocking_probes"),
    }


def _run_paths(results_dir: Path, run_id: str) -> tuple[Path, Path, Path]:
    if not run_id.startswith("eval_") or "/" in run_id or "\\" in run_id:
        raise ValueError(f"Invalid run ID: {run_id}")
    return (
        results_dir / f"{run_id}.jsonl",
        results_dir / f"{run_id}_summary.json",
        results_dir / f"{run_id}_manifest.json",
    )


def register_run(results_dir: Path, run_id: str, *, snapshots_dir: Path | None = None) -> dict[str, Any]:
    """Register or verify one run without ever rewriting its manifest."""
    root = results_dir.parent.parent
    detail_path, summary_path, manifest_path = _run_paths(results_dir, run_id)
    code_zip_path = results_dir / f"{run_id}_code.zip"
    if not detail_path.exists() and not summary_path.exists():
        raise FileNotFoundError(f"No artifacts for {run_id}")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    records = _read_records(detail_path) if detail_path.exists() else []
    case_ids = [record.get("id") for record in records]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError(f"Duplicate case IDs in {detail_path}")
    if summary is not None and (not detail_path.exists() or len(records) != summary.get("cases")):
        raise ValueError(f"Summary/detail case count differs for {run_id}")
    status = "complete" if summary is not None else "incomplete"

    snapshots: dict[str, Any] = {}
    if summary is not None and snapshots_dir is not None:
        for name, source, key, subdir, suffix in (
            ("evaluation_dataset", root / "evals/evaluations.json", "evaluation_dataset_sha256", "datasets", ".json"),
            ("database", root / "data/db/clinical_trials.duckdb", "database_sha256", "databases", ".duckdb"),
        ):
            expected = summary.get(key)
            if source.exists() and expected and file_sha256(source) == expected:
                saved = snapshot_file(source, snapshots_dir / subdir, suffix)
                snapshots[name] = _artifact(saved, root)

    manifest = {
        "history_version": HISTORY_VERSION,
        "run_id": run_id,
        "status": status,
        "kind": "rescore" if summary and summary.get("rescored_from") else "model_run",
        "timestamp_utc": summary.get("timestamp_utc") if summary else None,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "detail": _artifact(detail_path, root),
        "summary": _artifact(summary_path, root),
        "code_zip": _artifact(code_zip_path, root),
        "case_ids_sha256": hashlib.sha256(json.dumps(sorted(case_ids)).encode()).hexdigest(),
        "record_count": len(records),
        "model": summary.get("model") if summary else None,
        "quantization": summary.get("quantization") if summary else None,
        "runtime": summary.get("runtime") if summary else None,
        "model_runtime_configuration": summary.get("model_runtime_configuration") if summary else None,
        "model_sha256": summary.get("model_sha256") if summary else None,
        "prompt_version": summary.get("prompt_version") if summary else None,
        "schema_context_mode": summary.get("schema_context_mode") if summary else None,
        "embedding_retrieval_version": summary.get("embedding_retrieval_version") if summary else None,
        "embedding_corpus_sha256": summary.get("embedding_corpus_sha256") if summary else None,
        "embedding_model_id": summary.get("embedding_model_id") if summary else None,
        "embedding_encoding_version": summary.get("embedding_encoding_version") if summary else None,
        "embedding_model_revision": summary.get("embedding_model_revision") if summary else None,
        "retrieval_cache_sha256": summary.get("retrieval_cache_sha256") if summary else None,
        "max_repairs": summary.get("max_repairs") if summary else None,
        "reasoning_enabled": summary.get("reasoning_enabled") if summary else None,
        "intent_contract_version": summary.get("intent_contract_version") if summary else None,
        "question_timeout_seconds": summary.get("question_timeout_seconds") if summary else None,
        "prompt_sha256": summary.get("prompt_sha256") if summary else None,
        "scorer_version": summary.get("scorer_version") if summary else None,
        "scorer_sha256": summary.get("scorer_sha256") if summary else None,
        "evaluation_dataset_sha256": summary.get("evaluation_dataset_sha256") if summary else None,
        "database_sha256": summary.get("database_sha256") if summary else None,
        "code_commit": summary.get("code_commit") if summary else None,
        "rescored_from": summary.get("rescored_from") if summary else None,
        "snapshots": snapshots,
        "metrics": _summary_metrics(summary) if summary else None,
    }
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        keys_to_verify = ["detail", "summary", "case_ids_sha256", "record_count", "status"]
        if prior.get("code_zip") is not None:
            keys_to_verify.append("code_zip")
        for key in keys_to_verify:
            if prior.get(key) != manifest[key]:
                raise ValueError(f"Registered run artifact changed: {run_id} ({key})")
        for artifact in prior.get("snapshots", {}).values():
            saved = root / artifact["path"]
            if not saved.exists() or file_sha256(saved) != artifact["sha256"]:
                raise ValueError(f"Registered snapshot changed: {saved}")
        return prior
    with manifest_path.open("x", encoding="utf-8") as target:
        target.write(json.dumps(manifest, indent=2, default=str) + "\n")
    return manifest


def sync_history(results_dir: Path, *, snapshots_dir: Path | None = None) -> list[dict[str, Any]]:
    """Backfill manifests and verify all previously registered runs."""
    run_ids = {path.stem for path in results_dir.glob("eval_*.jsonl")}
    run_ids.update(path.name.removesuffix("_summary.json") for path in results_dir.glob("eval_*_summary.json"))
    manifests = []
    root = results_dir.parent.parent
    for run_id in sorted(run_ids):
        entry = dict(register_run(results_dir, run_id, snapshots_dir=snapshots_dir))
        summary_path = results_dir / f"{run_id}_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            entry["intent_contract_version"] = summary.get("intent_contract_version")
            entry["question_timeout_seconds"] = summary.get("question_timeout_seconds")
            entry["metrics"] = _summary_metrics(summary)
        # Reviews are completed after a run and are linked from the rebuildable
        # catalog without mutating the run's original checksum manifest.
        review_path = results_dir / f"{run_id}_error_register_v2.json"
        if not review_path.exists():
            review_path = results_dir / f"{run_id}_error_register.json"
        if review_path.exists():
            checksum_path = review_path.with_suffix(".json.sha256")
            if not checksum_path.exists() or json.loads(checksum_path.read_text())["sha256"] != file_sha256(review_path):
                raise ValueError(f"Error register failed checksum verification: {review_path}")
            review = json.loads(review_path.read_text(encoding="utf-8"))
            if review.get("run_id") != run_id:
                raise ValueError(f"Review run ID does not match: {review_path}")
            entry["error_register"] = _artifact(review_path, root)
            entry["semantic_review_counts"] = review["counts"]
        validity_path = results_dir / f"{run_id}_validity.json"
        if validity_path.exists():
            checksum_path = validity_path.with_suffix(".json.sha256")
            if not checksum_path.exists() or json.loads(checksum_path.read_text())["sha256"] != file_sha256(validity_path):
                raise ValueError(f"Validity annotation failed checksum verification: {validity_path}")
            validity = json.loads(validity_path.read_text(encoding="utf-8"))
            if validity.get("run_id") != run_id or not isinstance(validity.get("valid"), bool):
                raise ValueError(f"Invalid validity annotation: {validity_path}")
            entry["experiment_validity"] = validity
            entry["validity_annotation"] = _artifact(validity_path, root)
        counterfactual_path = results_dir / f"{run_id}_counterfactual_v1.json"
        if counterfactual_path.exists():
            checksum_path = counterfactual_path.with_suffix(".json.sha256")
            if not checksum_path.exists() or json.loads(checksum_path.read_text())["sha256"] != file_sha256(counterfactual_path):
                raise ValueError(f"Counterfactual report failed checksum verification: {counterfactual_path}")
            entry["counterfactual_report"] = _artifact(counterfactual_path, root)
            counterfactual = json.loads(counterfactual_path.read_text(encoding="utf-8"))
            entry["counterfactual_scores"] = {
                "matches": counterfactual["counterfactual_matches"],
                "cases": len(counterfactual["cases"]),
                "scorer_version": counterfactual["scorer_version"],
                "fixture_sha256": counterfactual["fixture_sha256"],
            }
        contract_audit_path = results_dir / f"{run_id}_contract_audit_v1.json"
        if contract_audit_path.exists():
            checksum_path = contract_audit_path.with_suffix(".json.sha256")
            if not checksum_path.exists() or json.loads(checksum_path.read_text())["sha256"] != file_sha256(contract_audit_path):
                raise ValueError(f"Contract audit failed checksum verification: {contract_audit_path}")
            audit = json.loads(contract_audit_path.read_text(encoding="utf-8"))
            if audit.get("run_id") != run_id:
                raise ValueError(f"Contract audit run ID does not match: {contract_audit_path}")
            entry["contract_audit"] = _artifact(contract_audit_path, root)
            entry["contract_audit_counts"] = {
                "rejected_attempts": audit["rejected_attempts"],
                "cases_with_rejection": audit["cases_with_rejection"],
                "possible_false_blocks": audit["possible_false_blocks"],
            }
        code_zip_path = results_dir / f"{run_id}_code.zip"
        if code_zip_path.exists():
            checksum_path = code_zip_path.with_suffix(".zip.sha256")
            if not checksum_path.exists() or json.loads(checksum_path.read_text())["sha256"] != file_sha256(code_zip_path):
                raise ValueError(f"Code ZIP failed checksum verification: {code_zip_path}")
            entry["code_zip"] = _artifact(code_zip_path, root)
        manifests.append(entry)
    catalog = {"history_version": HISTORY_VERSION, "runs": manifests}
    catalog_path = results_dir / "catalog.json"
    temporary = results_dir / f".catalog.{os.getpid()}.tmp"
    try:
        temporary.write_text(json.dumps(catalog, indent=2, default=str) + "\n", encoding="utf-8")
        os.replace(temporary, catalog_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifests


def sync_project_history(project_root: Path) -> list[dict[str, Any]]:
    """Verify legacy and current runs and rebuild their combined tracker catalog."""
    tracker_dir = project_root / "experiment-tracker"
    current_dir = tracker_dir / "results"
    current_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = project_root / "evals/results"
    manifests = []
    if legacy_dir.exists():
        manifests.extend(sync_history(legacy_dir, snapshots_dir=legacy_dir / "snapshots"))
    manifests.extend(sync_history(current_dir, snapshots_dir=current_dir / "snapshots"))
    manifests.sort(key=lambda item: item["run_id"])
    (tracker_dir / "catalog.json").write_text(
        json.dumps({"history_version": HISTORY_VERSION, "runs": manifests}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifests


def history_report(manifests: list[dict[str, Any]], *, report_dir: Path | None = None) -> str:
    """Build a chronological ledger with explicit comparability groups."""
    complete = [item for item in manifests if item["status"] == "complete" and item.get("experiment_validity", {}).get("valid", True)]
    incomplete = [item for item in manifests if item["status"] != "complete"]
    invalid = [item for item in manifests if item["status"] == "complete" and not item.get("experiment_validity", {}).get("valid", True)]
    lines = ["# Clinical trial evaluation history", "", "This report summarizes preserved evaluation runs and their comparable groups.", "", f"Registered runs: {len(manifests)} ({len(complete)} valid complete, {len(invalid)} invalid complete, {len(incomplete)} incomplete).", ""]
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for item in complete:
        key = (
            item.get("evaluation_dataset_sha256"), item.get("database_sha256"),
            item.get("scorer_version") or item.get("scorer_sha256"), item.get("case_ids_sha256"),
        )
        groups.setdefault(key, []).append(item)
    for number, (key, runs) in enumerate(groups.items(), start=1):
        lines.extend([
            f"## Comparable group {number}", "",
            f"Dataset `{key[0] or 'unknown'}`; database `{key[1] or 'unknown'}`; scorer identity `{key[2] or 'unknown'}`; case set `{key[3] or 'unknown'}`.",
            "", "| Run | Kind | Prompt | Generated | Safety | Schema | Executed | Correct rows | Counterfactual | Reviewed semantic | First pass | Repair | Attempts | Median latency | Code ZIP |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ])
        for item in runs:
            metrics = item["metrics"]
            def fraction(name: str) -> str:
                score = metrics.get(name) or {}
                return f"{score.get('passed', '?')}/{score.get('applicable', '?')}"
            review = item.get("semantic_review_counts")
            semantic = f"{review['reviewed_semantic_correct']}/{review['reference_cases']}" if review else "—"
            cf = item.get("counterfactual_scores")
            cf_score = f"{cf['matches']}/{cf['cases']}" if cf else "—"
            code_zip = item.get("code_zip")
            zip_name = Path(code_zip["path"]).name if code_zip else None
            zip_target = os.path.relpath(report_dir.parent / code_zip["path"], report_dir) if report_dir and code_zip else zip_name
            zip_link = f"[{zip_name}]({zip_target})" if zip_name else "—"
            lines.append(
                f"| {item['run_id']} | {item['kind']} | {item.get('prompt_version') or '?'} | "
                f"{fraction('sql_generated')} | {fraction('sql_safety')} | {fraction('schema_compliance')} | "
                f"{fraction('execution')} | {fraction('correct_result')} | {cf_score} | {semantic} | "
                f"{fraction('first_pass_accuracy')} | {fraction('repair_success')} | "
                f"{metrics.get('repair_attempts')} | {metrics.get('median_latency_seconds')} s | {zip_link} |"
            )
        lines.append("")
        known_scorer_hashes = {item["scorer_sha256"] for item in runs if item.get("scorer_sha256")}
        if len(known_scorer_hashes) > 1:
            lines.append("Scorer source hashes differ inside this version; verify scoring comparability before drawing conclusions.")
            lines.append("")
    if incomplete:
        lines.extend(["## Incomplete runs", "", "These JSONL files have no completed summary:", ""])
        lines.extend(f"- `{item['run_id']}`: {item['record_count']} saved records" for item in incomplete)
        lines.append("")
    if invalid:
        lines.extend(["## Invalid runs", "", "These artifacts are preserved but excluded from comparisons:", ""])
        lines.extend(f"- `{item['run_id']}`: {item['experiment_validity']['reason']}" for item in invalid)
        lines.append("")
    lines.append("Only runs with the same dataset, database, scorer, and case set are grouped for direct comparison. Rescores are labeled separately from model runs.")
    return "\n".join(lines) + "\n"
