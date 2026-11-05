"""Separate ExecuTorch portability case, deliberately outside serving-engine comparisons."""

import json
import re
import subprocess
import time
from pathlib import Path

from clinical_trial_mvp.evaluation.dataset import sha256_file
from clinical_trial_mvp.serving.portable import portable_text
from clinical_trial_mvp.tutorial.memory import PeakMemory


def observer_metrics(output):
    match = re.search(r"PyTorchObserver\s+(\{[^\n]+\})", output)
    stats = json.loads(match.group(1)) if match else {}
    def duration(start, end):
        if start not in stats or end not in stats:
            return None
        value = (stats[end] - stats[start]) / 1000
        if value < 0:
            raise ValueError("Invalid ExecuTorch timing order")
        return value
    return {"startup_seconds": duration("model_load_start_ms", "model_load_end_ms"),
            "single_request_seconds": duration("inference_start_ms", "inference_end_ms")}


def edge_portability(spec, *, execute=False, root=Path(".")):
    result = {"experiment": "ExecuTorch/edge", "runtime": "executorch", "hardware": spec.get("hardware", "local CPU"),
              "status": "plan_only", "evidence": "none", "quality": None, "serving": None,
              "memory": {"peak_bytes": None, "scope": "runner RSS"},
              "startup_seconds": None, "single_request_seconds": None,
              "footprint_bytes": None, "operator_fallback_count": None,
              "reason": "Export the supported small model and build the matching official runner first."}
    if not execute:
        return result
    root = Path(root)
    try:
        runner, model, tokenizer = [root / spec[k] for k in ("runner", "model", "tokenizer")]
        if not all(p.is_file() for p in (runner, model, tokenizer)):
            raise ValueError("ExecuTorch runner, PTE, or tokenizer is missing")
        command = [str(runner.resolve()), "--model_path=" + str(model.resolve()),
                   "--tokenizer_path=" + str(tokenizer.resolve()), "--temperature=0",
                   "--seq_len=128", "--prompt=" + spec.get("prompt", "Explain why edge inference can run offline.")]
        if spec.get("python"):
            interpreter = root / spec["python"]
            if not interpreter.is_file():
                raise ValueError("The isolated edge Python interpreter is missing")
            command.insert(0, str(interpreter.absolute()))
        started = time.perf_counter()
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            try:
                with PeakMemory(server_pid=process.pid) as memory:
                    output, _ = process.communicate(timeout=spec.get("timeout_seconds", 180))
            except BaseException:
                process.kill()
                process.wait()
                raise
        if process.returncode:
            raise ValueError("ExecuTorch runner failed; inspect its local build/model compatibility")
        result.update(observer_metrics(output))
        generation = re.search(r"EdgeGeneration\s+(\{[^\n]+\})", output)
        if generation:
            result["generation"] = json.loads(generation.group(1))
        result.update(status="completed", evidence="hardware", memory=memory.result(),
                      footprint_bytes=model.stat().st_size + tokenizer.stat().st_size,
                      wall_seconds=time.perf_counter() - started, model_sha256=sha256_file(model),
                      tokenizer_sha256=sha256_file(tokenizer), runner_sha256=sha256_file(runner),
                      reason="Startup/inference split is unknown if this runner emits no observer metrics.")
        report_path = root / spec.get("operator_report", "tutorial/edge_operators.local.json")
        if report_path.exists():
            report = json.loads(report_path.read_text())
            fallback, total = report.get("fallback_count"), report.get("operator_count")
            if (report.get("evidence") != "runtime_trace" or report.get("model_sha256") != result["model_sha256"]
                    or type(fallback) is not int or type(total) is not int or not 0 <= fallback <= total):
                raise ValueError("Operator fallback report must match this PTE and contain reviewed runtime trace counts")
            result["operator_fallback_count"] = fallback
        else:
            result["operator_fallback_note"] = "Unknown until ETDump/ETRecord inspection; delegation is not runtime fallback."
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        result.update(status="failed", evidence="none", reason=portable_text(exc))
    return result


def edge_table(result):
    def fmt(key, scale=1):
        value = result.get(key)
        return "—" if value is None else f"{value / scale:.3f}"
    peak = result.get("memory", {}).get("peak_bytes")
    peak_text = "—" if peak is None else f"{peak / 1024**2:.1f}"
    return ("| Status | Model load s | Model+tokenizer MiB | Single request s | Peak runner RSS MiB | Operator fallback count |\n"
            "|---|---|---|---|---|---|\n"
            f"| {result['status']} | {fmt('startup_seconds')} | {fmt('footprint_bytes', 1024**2)} | "
            f"{fmt('single_request_seconds')} | {peak_text} | {fmt('operator_fallback_count')} |")
