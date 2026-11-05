"""Workload fairness, measured intervals, model provenance, and CLI tests."""

import copy
import importlib.util
import json
import threading
from pathlib import Path

import pytest

from clinical_trial_mvp.serving.benchmark import (
    Workload, compare, prepare, run_serving, serving_summary, snapshot_fingerprint,
)
from clinical_trial_mvp.serving.client import Sample
from clinical_trial_mvp.serving.launch import launch_command

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("kwargs", [{"repeats": 0}, {"warmups": -1}, {"concurrency": 0},
                                     {"mode": "quality", "concurrency": 2}, {"profile": "wrong"},
                                     {"max_repairs": 2}, {"e2e_slo_seconds": float("nan")}])
def test_invalid_workloads(kwargs):
    with pytest.raises(ValueError):
        Workload(**kwargs)


def test_serving_excludes_policy_cases_quality_keeps_all_cases():
    serving, _, _, _ = prepare(ROOT, Workload())
    quality, _, _, _ = prepare(ROOT, Workload(mode="quality"))
    assert len(serving) == 26 and len(quality) == 30
    with pytest.raises(ValueError, match="No cases"):
        prepare(ROOT, Workload(), ["eval_016"])
    with pytest.raises(ValueError, match="Unknown cases"):
        prepare(ROOT, Workload(), ["typo"])


def test_frozen_retrieved_prompts_and_fingerprints_are_repeatable():
    a = prepare(ROOT, Workload(profile="retrieved"), ["eval_001"])
    b = prepare(ROOT, Workload(profile="retrieved"), ["eval_001"])
    full = prepare(ROOT, Workload(), ["eval_001"])
    assert a == b
    assert a[3]["cache_sha256"] and a[3]["prompts_sha256"] != full[3]["prompts_sha256"]


def test_summary_counts_failures_slos_and_wall_clock_throughput():
    ok = Sample(success=True, ttft_seconds=1, e2e_seconds=2, output_tokens=10).to_dict()
    slow = Sample(success=True, ttft_seconds=6, e2e_seconds=9, output_tokens=20).to_dict()
    failed = Sample(error="reset", output_tokens=500).to_dict()
    summary = serving_summary([ok, slow, failed], 10, Workload())
    assert summary["requests"] == 3 and summary["failed"] == 1
    assert summary["slo_successful"] == 1
    assert summary["requests_per_second"] == .2 and summary["output_tokens_per_second"] == 3
    assert summary["ttft_p50_seconds"] == 3.5
    assert serving_summary([Sample(success=True).to_dict()], 1, Workload())["output_tokens_per_second"] is None
    assert serving_summary([failed], 1, Workload())["ttft_p95_seconds"] is None


def test_concurrent_load_and_warmup_exclusion():
    barrier = threading.Barrier(2, timeout=5)
    calls = []
    class Client:
        def complete(self, messages):
            calls.append(messages)
            if len(calls) > 1:  # One serial warmup, then two actual workers.
                barrier.wait()
            return Sample(success=True, content="ok", ttft_seconds=.1, e2e_seconds=1, output_tokens=5)
    emitted = []
    ticks = iter([100, 102])
    summary = run_serving(Client(), [{"id": "a"}, {"id": "b"}], {"a": "A", "b": "B"},
                          Workload(repeats=1, concurrency=2), emitted.append, clock=lambda: next(ticks))
    assert len(calls) == 3 and len(emitted) == 2
    assert summary["measurement_seconds"] == 2 and summary["requests_per_second"] == 1


def test_warmup_failure_aborts_measurements():
    class Client:
        def complete(self, messages):
            return Sample(error="unavailable")
    with pytest.raises(ValueError, match="Warmup failed"):
        run_serving(Client(), [{"id": "a"}], {"a": "a"}, Workload(), lambda r: pytest.fail("unexpected record"))


def manifests():
    a = {"runtime": "vllm", "status": "completed", "run_id": "a", "summary": {}, "server": {},
         "contract": {"workload": {"mode": "serving"}, "model_sha256": "same"}}
    b = copy.deepcopy(a)
    b.update(runtime="sglang", run_id="b")
    return a, b


def test_comparison_accepts_matched_pair():
    result = compare(*manifests())
    assert set(result["runs"]) == {"vllm", "sglang"}


@pytest.mark.parametrize("mutation", ["same_engine", "failed", "changed_model", "changed_workload"])
def test_comparison_rejects_invalid_pairs(mutation):
    a, b = manifests()
    if mutation == "same_engine":
        b["runtime"] = "vllm"
    elif mutation == "failed":
        b["status"] = "failed"
    elif mutation == "changed_model":
        b["contract"]["model_sha256"] = "other"
    else:
        b["contract"]["workload"]["concurrency"] = 8
    with pytest.raises(ValueError):
        compare(a, b)


def test_snapshot_hash_covers_weights_and_template_but_ignores_download_cache(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    template = tmp_path / "chat_template.jinja"
    template.write_text("first")
    first = snapshot_fingerprint(tmp_path)
    template.write_text("second")
    second = snapshot_fingerprint(tmp_path)
    assert first != second
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache/noise.json").write_text("noise")
    assert snapshot_fingerprint(tmp_path) == second


def test_snapshot_without_weights_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="weights"):
        snapshot_fingerprint(tmp_path)


def test_engine_commands_match_dtype_context_tp_and_disable_prefix_cache():
    v = launch_command("vllm", "/snapshot", context=4096, tp=2)
    s = launch_command("sglang", "/snapshot", context=4096, tp=2)
    assert v[v.index("--max-model-len") + 1] == s[s.index("--context-length") + 1] == "4096"
    assert v[v.index("--tensor-parallel-size") + 1] == s[s.index("--tp-size") + 1] == "2"
    assert "--no-enable-prefix-caching" in v and "--disable-radix-cache" in s
    assert "--no-enable-prefix-caching" not in launch_command("vllm", "/snapshot", prefix_cache=True)


def load_cli():
    spec = importlib.util.spec_from_file_location("mvp3_cli", ROOT / "scripts/serving/compare_vllm_sglang.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_dry_run_needs_no_server_or_gpu_and_writes_no_results(tmp_path, capsys):
    result = load_cli().main(["run", "--runtime", "vllm", "--dry-run", "--output", str(tmp_path / "runs")])
    assert result == 0
    assert json.loads(capsys.readouterr().out)["status"] == "plan_only"
    assert not (tmp_path / "runs").exists()


def test_cli_live_run_requires_metadata(tmp_path):
    with pytest.raises(SystemExit):
        load_cli().main(["run", "--runtime", "sglang", "--output", str(tmp_path)])
    assert list(tmp_path.iterdir()) == []


def test_cli_writes_checksummed_runs_then_compares_them(tmp_path, monkeypatch):
    module = load_cli()
    class Client:
        def __init__(self, base_url, model, generation):
            self.base_url, self.model = base_url, model
        def complete(self, messages):
            return Sample(success=True, content="SELECT 1", ttft_seconds=.1, e2e_seconds=.2, output_tokens=3)
    monkeypatch.setattr(module, "OpenAIClient", Client)
    manifests_saved = []
    for runtime in ("vllm", "sglang"):
        metadata = tmp_path / (runtime + ".json")
        metadata.write_text(json.dumps({"runtime": runtime, "engine_version": "test-only", "model_sha256": "same",
                                        "hardware": "mock-gpu", "launch_settings": {"context": 8192},
                                        "command": ["mock"], "cuda_visible_devices": "all",
                                        "model": "clinical-trial-mvp3"}))
        module.main(["run", "--runtime", runtime, "--server-metadata", str(metadata),
                     "--case", "eval_001", "--repeats", "1", "--output", str(tmp_path / "runs")])
        manifests_saved.append(next((tmp_path / "runs").glob(f"*_{runtime}/manifest.json")))
    output = tmp_path / "comparison.json"
    assert module.main(["compare", *map(str, manifests_saved), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["scope"] == "serving"
    # Editing detailed evidence invalidates the comparison.
    (manifests_saved[0].parent / "samples.jsonl").write_text("tampered")
    with pytest.raises(SystemExit):
        module.main(["compare", *map(str, manifests_saved), "--output", str(tmp_path / "bad.json")])


def test_matrix_covers_both_profiles_and_all_concurrency_levels():
    spec = importlib.util.spec_from_file_location("mvp3_matrix", ROOT / "scripts/serving/run_serving_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    commands = list(module.matrix_commands("vllm"))
    assert len(commands) == 8
    combinations = {(c[c.index("--mode") + 1], c[c.index("--profile") + 1],
                     c[c.index("--concurrency") + 1]) for c in commands}
    assert combinations == {(mode, profile, str(n)) for mode, levels in
                            (("serving", [1, 2, 4]), ("quality", [1]))
                            for profile in ("full", "retrieved") for n in levels}
