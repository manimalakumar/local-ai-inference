import hashlib
import json
from pathlib import Path

import pytest

from clinical_trial_mvp.agent.analyst import DEFAULT_QUESTION_SQL
from clinical_trial_mvp.serving.client import Generation, Sample
from clinical_trial_mvp.tutorial.harness import (
    findings, load_configurations, pair_status, semantic_scores, serve_and_benchmark, summary_markdown, validate_provenance,
)
from clinical_trial_mvp.tutorial.memory import PeakMemory

ROOT = Path(__file__).resolve().parents[2]
PLAN = {"population": "Enrolled or Randomized subjects", "measure": "count subjects", "grouping": "treatment arm",
        "time_window": "none", "filters": "screening_status in Enrolled or Randomized", "ranking": "none",
        "needs_clarification": False, "clarification_question": ""}


def spec():
    return {**load_configurations(ROOT)["rtx_vllm"], "warmups": 0, "repeats": 1}


def server():
    return {"runtime": "vllm", "engine_version": "unit-test", "hardware": "fake GPU", "model_sha256": "a" * 64,
            "evidence": "hardware", "launch_settings": {"context": 8192, "dtype": "bfloat16", "prefix_cache": False}}


def test_plan_does_not_construct_a_client_or_emit_measurements(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("Plan must not construct an inference client")
    monkeypatch.setattr("clinical_trial_mvp.tutorial.harness.serving_adapter", unexpected)
    result = serve_and_benchmark(spec(), root=ROOT, case_ids=["eval_001"])
    assert result["status"] == "plan_only" and result["quality"] is None
    assert result["memory"]["peak_bytes"] is None


def test_common_harness_uses_existing_clinical_path_and_excludes_fake_hardware_results():
    class Fake:
        generation = Generation()
        def complete(self, messages, schema=None):
            content = PLAN if schema and "population" in schema.get("properties", {}) else {"sql": DEFAULT_QUESTION_SQL, "rationale": ""}
            return Sample(success=True, content=json.dumps(content), prompt_tokens=12,
                          output_tokens=10, ttft_seconds=.1, e2e_seconds=.2)
    result = serve_and_benchmark(spec(), execute=True, root=ROOT, case_ids=["eval_001"],
        client=Fake(), server=server(), memory=PeakMemory(reader=lambda: 1024))
    assert result["status"] == "completed"
    assert result["quality"]["result_equivalent"] == 1 and result["quality"]["sql_executed"] == 1
    assert result["serving"]["mean_prompt_tokens"] == 12
    assert result["quality"]["semantic"]["total"] == 1
    assert result["evidence"] == "test_or_unverified"
    assert "1/1 match" not in summary_markdown([result])


@pytest.mark.parametrize("change", [{"runtime": "sglang"}, {"model_sha256": "invalid"},
                                    {"launch_settings": {"dtype": "float16", "context": 8192}}])
def test_provenance_rejects_wrong_engine_model_or_precision(change):
    with pytest.raises(ValueError):
        validate_provenance(spec(), {**server(), **change})


def test_optimization_setting_must_match_the_running_server():
    with pytest.raises(ValueError, match="quantization"):
        validate_provenance({**spec(), "quantization": "awq"}, server())
    with pytest.raises(ValueError, match="prefix_cache"):
        validate_provenance({**spec(), "prefix_cache": True}, server())


def test_semantic_review_is_exact_and_independent_of_result_match():
    sql = "SELECT 1"
    records = [{"case_id": "a", "attempts": [{"sql": sql}], "scores": {"result_equivalence": True}}]
    review = {"case_id": "a", "sql_sha256": hashlib.sha256(sql.encode()).hexdigest(),
              "reviewer": "reviewer", "reviewed_at": "2026-10-25", "correct": False, "rationale": "Fixture: result match alone does not prove intent."}
    assert semantic_scores(records, [review]) == {"correct": 0, "reviewed": 1, "total": 1}
    assert semantic_scores(records, []) == {"correct": 0, "reviewed": 0, "total": 1}
    with pytest.raises(ValueError):
        semantic_scores(records, [review, review])


def test_memory_unknown_is_not_zero_and_reader_failures_remain_unknown():
    with PeakMemory() as missing:
        pass
    assert missing.result()["peak_bytes"] is None
    def fail():
        raise OSError("not available")
    with PeakMemory(reader=fail) as failed:
        pass
    assert failed.result()["peak_bytes"] is None and failed.result()["error"] == "OSError"
    with PeakMemory(reader=lambda: 2048) as observed:
        pass
    assert observed.result()["peak_bytes"] == 2048 and observed.result()["sample_count"] >= 2
