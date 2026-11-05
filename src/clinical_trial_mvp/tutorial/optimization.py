"""MVP4 vLLM factorial study, reusing the existing clinical tutorial harness."""

from __future__ import annotations

import json
import shlex
import platform
from datetime import datetime, timezone
from dataclasses import replace, asdict
from pathlib import Path

from clinical_trial_mvp.evaluation.dataset import sha256_file
from clinical_trial_mvp.serving.benchmark import digest, snapshot_fingerprint
from clinical_trial_mvp.tutorial.harness import ROOT, serve_and_benchmark, validate_provenance
from clinical_trial_mvp.tutorial.cache_study import measure_requests, ready, messages, sql_score
from clinical_trial_mvp.serving.client import OpenAIClient, Generation
from clinical_trial_mvp.serving.portable import portable_value

VARIANTS = ("fp16_cache_off", "fp16_cache_on", "awq_w4a16_cache_off", "awq_w4a16_cache_on")


def preflight(spec, *, model_id, base_revision, awq_source, execute=False):
    import importlib.metadata
    import subprocess
    result = {"python": platform.python_version(), "model_id": model_id,
              "base_revision": base_revision, "awq_source": awq_source,
              "context": spec["context"], "endpoint": spec["base_url"],
              "gpu": "pending", "endpoint_status": "not checked (offline)"}
    for name in ("torch", "vllm"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed in notebook environment"
    if execute:
        try:
            result["gpu"] = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader"], text=True, timeout=10).strip()
            result["cuda_driver_capability"] = subprocess.check_output(["nvidia-smi"], text=True, timeout=10).splitlines()[2]
        except (OSError, subprocess.SubprocessError, IndexError) as exc:
            result["gpu"] = str(exc)
        try:
            ready(OpenAIClient(spec["base_url"], spec["model"]))
            result["endpoint_status"] = "ready (health 200)"
        except (OSError, ValueError) as exc:
            result["endpoint_status"] = str(exc)
    return result


def configurations(*, run_tag="run01", hardware="GPU host (verify SKU)", context=8192,
                   repeats=5, concurrency=1):
    if not run_tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in run_tag):
        raise ValueError("Use a simple unique run_tag: letters, numbers, underscore, hyphen")
    if concurrency != 1:
        raise ValueError("Cold/warm requests require concurrency=1")
    return {name: {
        "experiment": "MVP4/Optimization/" + name, "runtime": "vllm", "host": "cuda",
        "hardware": hardware, "model": "clinical-trial-mvp3", "base_url": "http://127.0.0.1:8000",
        "context": context, "dtype": "float16", "quantization": "awq" if name.startswith("awq") else None,
        "prefix_cache": name.endswith("_on"), "profile": "full", "concurrency": concurrency,
        "repeats": repeats, "warmups": 1, "max_tokens": 1024,
        "cache_regime": "reset_cold_reset_distinct_question_prime_warm_v2",
        "provenance": f"experiment-tracker/mvp4/provenance/{run_tag}-{name}.json",
    } for name in VARIANTS}


def checkpoint_pair(fp16, awq, *, base_revision):
    """Validate local artifacts; revision is an explicit lineage attestation, not inferred."""
    if not base_revision or base_revision.startswith("SET_"):
        raise ValueError("Record the exact common base model/revision and AWQ derivation source")
    fp16, awq = Path(fp16), Path(awq)
    configs = [json.loads((p / "config.json").read_text()) for p in (fp16, awq)]
    if configs[0].get("quantization_config"):
        raise ValueError("FP16 baseline must have unquantized weights")
    quant = configs[1].get("quantization_config", {})
    if quant.get("quant_method", "").lower() != "awq" or quant.get("bits") != 4:
        raise ValueError("AWQ W4A16 requires quant_method=awq and bits=4")
    architecture = ("model_type", "architectures", "hidden_size", "num_hidden_layers", "vocab_size")
    for key in architecture:
        if not configs[0].get(key) or configs[0].get(key) != configs[1].get(key):
            raise ValueError("Checkpoint architecture mismatch/missing: " + key)
    def tokenizer_files(folder):
        return {str(p.relative_to(folder)): sha256_file(p) for p in sorted(folder.rglob("*"))
                if p.is_file() and (p.name.startswith(("tokenizer", "special_tokens", "added_tokens", "chat_template"))
                                    or p.name in {"vocab.json", "merges.txt", "vocab.txt", "spiece.model"})
                and ".cache" not in p.relative_to(folder).parts}
    left, right = tokenizer_files(fp16), tokenizer_files(awq)
    if not left or left != right:
        raise ValueError("Tokenizer/chat template artifacts must match exactly")
    # Weight fingerprints are deliberately different across precisions.
    return {"base_revision_attestation": base_revision, "tokenizer_sha256": digest(left),
            "fp16_sha256": snapshot_fingerprint(fp16), "awq_sha256": snapshot_fingerprint(awq),
            "awq_config": quant}


def launch_preview(spec, snapshot, *, python=".venv-vllm/bin/python"):
    command = [python, "scripts/serving/compare_vllm_sglang.py", "launch", "--runtime", "vllm",
               "--snapshot", str(snapshot), "--dtype", "float16", "--context", str(spec["context"]),
               "--metadata", spec["provenance"], "--execute"]
    if spec["quantization"]:
        command += ["--quantization", spec["quantization"]]
    if spec["prefix_cache"]:
        command += ["--prefix-cache"]
    return shlex.join(command)


def validated_server(spec, pair, root):
    if not pair:
        raise ValueError("Validate checkpoint_pair before live execution")
    server = json.loads((Path(root) / spec["provenance"]).read_text())
    validate_provenance(spec, server)
    expected = pair["awq_sha256" if spec["quantization"] == "awq" else "fp16_sha256"]
    if server["model_sha256"] != expected:
        raise ValueError("Server checkpoint fingerprint differs from validated pair")
    if platform.system() != "Linux":
        raise ValueError("Live execution requires the Linux GPU host")
    if not server.get("gpu_uuid"):
        raise ValueError("Record the actual GPU UUID for GPU memory measurement")
    return server


def save_record(root, category, result):
    folder = Path(root) / "experiment-tracker/mvp4" / category
    folder.mkdir(parents=True, exist_ok=True)
    result = portable_value(result)
    with (folder / (result["run_id"] + ".json")).open("x") as output:
        json.dump(result, output, indent=2, default=str)
        output.write("\n")


def awq_smoke(spec, pair=None, *, execute=False, root=ROOT):
    """One real SQL generation/execution verifies AWQ before any factorial runs."""
    result = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
              "status": "plan_only", "evidence": "none", "checkpoint_pair": pair,
              "configuration": spec}
    if not execute:
        return result
    if spec.get("quantization") != "awq" or spec.get("prefix_cache"):
        raise ValueError("Smoke-test AWQ W4A16 with cache off first")
    try:
        server = validated_server(spec, pair, root)
        result["server"] = server
        client = OpenAIClient(spec["base_url"], spec["model"], Generation(max_tokens=spec["max_tokens"]))
        ready(client)
        from clinical_trial_mvp.config import Settings
        from clinical_trial_mvp.serving.benchmark import Workload, prepare
        from clinical_trial_mvp.data.db import execute_readonly_sql
        cases, _, _, _ = prepare(Path(root), Workload(profile="full"), ["eval_001"])
        settings = replace(Settings.from_env(), project_root=Path(root), db_path=Path(root) / "data/db/clinical_trials.duckdb")
        reference = execute_readonly_sql(cases[0]["reference_sql"], settings)
        if not reference.ok:
            raise ValueError("Smoke reference SQL failed")
        sample = client.complete(messages(cases[0]["question"]))
        result.update(sample=sample.to_dict(), **sql_score(sample, cases[0], asdict(reference), settings))
        if not sample.success or not result["attempts"][0]["execution_success"]:
            raise ValueError("AWQ smoke failed: completion or executable SQL unavailable")
        result.update(status="completed", evidence=server.get("evidence", "none"))
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        result.update(status="failed", reason=str(exc))
    save_record(root, "smoke", result)
    return portable_value(result)


def check_smoke(smoke, pair, server):
    if not smoke or smoke.get("status") != "completed" or smoke.get("evidence") != "hardware":
        raise ValueError("Complete the AWQ compatibility smoke test before all four variants")
    if smoke.get("checkpoint_pair") != pair:
        raise ValueError("Smoke test belongs to a different checkpoint pair")
    for key in ("engine_version", "hardware", "gpu_uuid", "torch_version", "cuda_version", "cuda_visible_devices"):
        if not server.get(key) or smoke.get("server", {}).get(key) != server[key]:
            raise ValueError("AWQ smoke environment differs: " + key)


def run_variant(spec, pair=None, *, smoke=None, execute=False, root=ROOT):
    """No server access by default. Hardware results retain exact checkpoint lineage."""
    root = Path(root)
    if execute:
        server = validated_server(spec, pair, root)
        check_smoke(smoke, pair, server)
        client = OpenAIClient(spec["base_url"], spec["model"], Generation(max_tokens=spec["max_tokens"]))
        try:
            ready(client)
            study = measure_requests(spec, client, root=root, gpu_uuid=server["gpu_uuid"])
        except (OSError, ValueError, RuntimeError, KeyError) as exc:
            result = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"),
                      "experiment": spec["experiment"], "configuration": spec, "server": server,
                      "checkpoint_pair": pair, "status": "failed", "evidence": "none", "reason": str(exc)}
            save_record(root, "runs", result)
            return portable_value(result)
    else:
        server = None
    result = serve_and_benchmark(spec, execute=execute, root=root, server=server)
    result["checkpoint_pair"] = pair
    result["cache_study"] = study if execute else None
    result["awq_smoke_run_id"] = smoke["run_id"] if execute else None
    # The inherited harness also retains its raw tutorial record. This dedicated copy
    # includes the precision lineage needed to compare different weight fingerprints.
    if execute:
        save_record(root, "runs", result)
    return result


def compare_pair(left, right, *, factor, phase="warm"):
    """Reject confounded pairs before calculating latency ratios."""
    if factor not in {"prefix_cache", "quantization"}:
        raise ValueError("Choose prefix_cache or quantization")
    if phase not in {"cold", "warm"}:
        raise ValueError("Compare cold or warm separately")
    for result in (left, right):
        if result.get("status") != "completed" or result.get("evidence") != "hardware":
            raise ValueError("Two completed hardware runs are required")
        if not result.get("checkpoint_pair"):
            raise ValueError("Missing checkpoint lineage")
        spec = result["configuration"]
        if spec.get("dtype") != "float16" or spec.get("quantization") not in {None, "awq"}:
            raise ValueError("This study requires FP16 or AWQ W4A16")
        validate_provenance(spec, result["server"])
        expected = result["checkpoint_pair"]["awq_sha256" if spec["quantization"] else "fp16_sha256"]
        if result["server"]["model_sha256"] != expected:
            raise ValueError("Checkpoint fingerprint does not match precision lineage")
        serving = result.get("serving") or {}
        if serving.get("failed") != 0 or not serving.get("requests"):
            raise ValueError("Serving requests failed or are missing")
        study = result.get("cache_study") or {}
        selected = study.get("phases", {}).get(phase if spec["prefix_cache"] else "cache_off")
        if not selected or selected.get("failed") != 0:
            raise ValueError("Missing successful cold/warm measurement")
    for key in ("inputs", "database_sha256", "evaluation_date", "checkpoint_pair"):
        if left.get(key) != right.get(key) or left.get(key) is None:
            raise ValueError("Unpaired: " + key)
    a, b = left["configuration"], right["configuration"]
    ignored = {"experiment", "provenance", factor}
    if {k: v for k, v in a.items() if k not in ignored} != {k: v for k, v in b.items() if k not in ignored}:
        raise ValueError("Other configuration settings differ")
    if a[factor] == b[factor]:
        raise ValueError("Selected factor did not change")
    for key in ("engine_version", "hardware", "gpu_uuid", "torch_version", "cuda_version", "cuda_visible_devices"):
        if not left["server"].get(key) or left["server"].get(key) != right["server"].get(key):
            raise ValueError("Unpaired server: " + key)
    launch_a, launch_b = (dict(r["server"]["launch_settings"]) for r in (left, right))
    launch_a.pop(factor, None)
    launch_b.pop(factor, None)
    if launch_a != launch_b:
        raise ValueError("Other server launch settings differ")
    if factor == "prefix_cache" and left["server"]["model_sha256"] != right["server"]["model_sha256"]:
        raise ValueError("Cache comparison requires identical weights")
    if left["cache_study"]["shared_prefix_sha256"] != right["cache_study"]["shared_prefix_sha256"]:
        raise ValueError("Shared prefixes differ")
    def ratio(key):
        summaries = [r["cache_study"]["phases"][phase if r["configuration"]["prefix_cache"] else "cache_off"]
                     for r in (left, right)]
        x, y = (s.get(key) for s in summaries)
        return x / y if x is not None and y is not None and y > 0 else None
    selected_phases = [r["cache_study"]["phases"][phase if r["configuration"]["prefix_cache"] else "cache_off"]
                       for r in (left, right)]
    return {"factor": factor, "phase": phase, "left": left["experiment"], "right": right["experiment"],
            "ttft_speedup_left_over_right": ratio("ttft_p50_seconds"),
            "latency_speedup_left_over_right": ratio("e2e_p50_seconds"),
            "result_matches_left_right": [r["quality"]["result_equivalent"] for r in (left, right)],
            "reference_scored_left_right": [r["quality"]["reference_scored"] for r in (left, right)],
            "phase_result_matches_left_right": [s["result_equivalent"] for s in selected_phases],
            "phase_reference_scored_left_right": [s["reference_scored"] for s in selected_phases],
            "note": "Cold/warm request timings exclude readiness, startup, reset, priming and SQL execution. "
                    "Clinical agent latency and correctness are reported separately."}
