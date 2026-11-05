"""Verify the measurement protocol with real DuckDB scoring and fake inference."""
import copy
import json
from dataclasses import replace

import pytest

from clinical_trial_mvp.tutorial import cache_study as study
from clinical_trial_mvp.tutorial import optimization as opt
from clinical_trial_mvp.tutorial.harness import ROOT, TUTORIAL_CASES
from clinical_trial_mvp.serving.benchmark import Workload, prepare
from clinical_trial_mvp.serving.client import Sample


class Memory:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def result(self):
        return {"peak_bytes": 1024**3, "scope": "test"}


def fake_client(events):
    cases, _, _, _ = prepare(ROOT, Workload(profile="full"), TUTORIAL_CASES)
    class Client:
        def complete(self, messages, schema=None):
            case = next(c for c in cases if study.messages(c["question"]) == messages)
            events.append(("request", case["id"]))
            return Sample(success=True, content=json.dumps({"sql": case["reference_sql"], "rationale": "test"}),
                          ttft_seconds=.1, e2e_seconds=.5, prompt_tokens=100, output_tokens=5)
    return Client()


def test_shared_schema_precedes_all_unique_questions():
    cases, _, _, _ = prepare(ROOT, Workload(profile="full"), TUTORIAL_CASES)
    prompts = [study.messages(c["question"]) for c in cases]
    assert len({m[0]["content"] for m in prompts}) == 1
    assert len({m[1]["content"].split("Question:\n")[0] for m in prompts}) == 1
    assert all(study.SCHEMA_PROMPT in m[1]["content"] for m in prompts)
    assert len({m[1]["content"] for m in prompts}) == 4


def test_cold_warm_are_separate_identical_targets_with_distinct_prime(monkeypatch):
    events = []
    client = fake_client(events)
    def reset():
        events.append(("reset",))
        return {"success": True}
    ticks = iter(range(100))
    spec = opt.configurations(repeats=1)["fp16_cache_on"]
    result = study.measure_requests(spec, client, root=ROOT, reset=reset,
                                    memory_factory=Memory, clock=lambda: next(ticks))
    assert events[0] == ("request", "eval_001")  # excluded kernel warmup
    assert events[1:6] == [("reset",), ("request", "eval_001"), ("reset",),
                           ("request", "eval_002"), ("request", "eval_001")]
    assert len(result["records"]) == 8 and len(result["priming_records"]) == 4
    assert all(p["target"] != p["prime_case"] for p in result["priming_records"])
    assert result["startup_in_request_timing"] is False
    for phase in ("cold", "warm"):
        summary = result["phases"][phase]
        assert summary["requests"] == 4
        assert summary["result_equivalent"] == summary["sql_executed"] == 4
        assert summary["measurement_seconds"] == 4  # only four target intervals
        assert summary["ttft_p50_seconds"] == .1
        assert summary["e2e_p50_seconds"] == .5
        assert summary["tpot_p50_seconds"] == .1
        assert summary["peak_gpu_bytes"] == 1024**3
    text = study.phase_table({"fp16_cache_on": {"cache_study": result}})
    assert "cold" in text and "warm" in text and "4/4" in text


def test_cache_off_never_resets_or_primes():
    events = []
    spec = opt.configurations(repeats=1)["fp16_cache_off"]
    result = study.measure_requests(spec, fake_client(events), root=ROOT,
        reset=lambda: pytest.fail("Cache off should not reset"), memory_factory=Memory)
    assert set(result["phases"]) == {"cache_off"}
    assert len(events) == 5 and result["priming_records"] == []


@pytest.mark.parametrize("response", [None, {}, {"success": False}, {"success": "true"}])
def test_legacy_or_failed_reset_cannot_label_a_cold_request(monkeypatch, response):
    monkeypatch.setattr(study, "control", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="Cold cache unverified"):
        study.reset_prefix(object())


def test_confirmed_reset_calls_only_control_api(monkeypatch):
    called = []
    monkeypatch.setattr(study, "control", lambda client, route, **kw: called.append((route, kw)) or {"success": True})
    assert study.reset_prefix(object())["success"] is True
    assert called == [("/reset_prefix_cache", {"method": "POST"})]


def test_failed_reset_aborts_before_measured_target():
    events = []
    with pytest.raises(ValueError, match="Reset did not confirm"):
        study.measure_requests(opt.configurations(repeats=1)["fp16_cache_on"], fake_client(events),
                               root=ROOT, reset=lambda: {"success": False}, memory_factory=Memory)
    assert len(events) == 1  # only excluded warmup


def test_sql_mismatch_and_invalid_output_are_not_counted_correct():
    from clinical_trial_mvp.config import Settings
    from clinical_trial_mvp.data.db import execute_readonly_sql
    from dataclasses import asdict
    settings = replace(Settings.from_env(), project_root=ROOT, db_path=ROOT / "data/db/clinical_trials.duckdb")
    case = prepare(ROOT, Workload(profile="full"), ["eval_001"])[0][0]
    reference = asdict(execute_readonly_sql(case["reference_sql"], settings))
    for content in ('{"sql":"SELECT 1"}', 'invalid JSON', '{"sql":"DROP TABLE subjects"}'):
        scored = study.sql_score(Sample(success=True, content=content), case, reference, settings)
        assert scored["scores"]["result_equivalence"] is False


def test_smoke_gate_blocks_all_variants_without_matching_hardware():
    pair = {"fp16_sha256": "a" * 64, "awq_sha256": "b" * 64}
    server = {k: "same" for k in ("engine_version", "hardware", "gpu_uuid", "torch_version",
                                  "cuda_version", "cuda_visible_devices")}
    smoke = {"status": "completed", "evidence": "hardware", "checkpoint_pair": pair, "server": server}
    opt.check_smoke(smoke, pair, server)
    for bad in (None, {**smoke, "status": "failed"}, {**smoke, "evidence": "test_or_unverified"},
                {**smoke, "checkpoint_pair": {}}, {**smoke, "server": {**server, "gpu_uuid": "other"}}):
        with pytest.raises(ValueError):
            opt.check_smoke(bad, pair, server)


def test_awq_smoke_performs_real_sql_check_before_saving(monkeypatch, tmp_path):
    events, saved = [], []
    server = {"evidence": "hardware", "gpu_uuid": "GPU-test"}
    monkeypatch.setattr(opt, "validated_server", lambda *a: server)
    monkeypatch.setattr(opt, "ready", lambda client: events.append(("ready",)))
    monkeypatch.setattr(opt, "OpenAIClient", lambda *a: fake_client(events))
    monkeypatch.setattr(opt, "save_record", lambda root, category, result: saved.append(copy.deepcopy(result)))
    result = opt.awq_smoke(opt.configurations()["awq_w4a16_cache_off"], {"test": True}, execute=True, root=ROOT)
    assert result["status"] == "completed"
    assert result["attempts"][0]["execution_success"]
    assert result["scores"]["result_equivalence"] is True
    assert events == [("ready",), ("request", "eval_001")]
    assert saved[0]["status"] == "completed"


def test_failed_awq_smoke_is_saved_and_cannot_open_gate(monkeypatch):
    saved = []
    monkeypatch.setattr(opt, "validated_server", lambda *a: {"evidence": "hardware"})
    class FailedClient:
        def complete(self, *a, **kw):
            return Sample(success=False, error="unsupported AWQ kernel")
    monkeypatch.setattr(opt, "OpenAIClient", lambda *a: FailedClient())
    monkeypatch.setattr(opt, "ready", lambda client: None)
    monkeypatch.setattr(opt, "save_record", lambda root, category, result: saved.append(copy.deepcopy(result)))
    result = opt.awq_smoke(opt.configurations()["awq_w4a16_cache_off"], {"test": True}, execute=True, root=ROOT)
    assert result["status"] == "failed" and saved[0]["sample"]["error"] == "unsupported AWQ kernel"
    with pytest.raises(ValueError):
        opt.check_smoke(result, {"test": True}, {})
