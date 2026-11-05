"""Exercise actual DuckDB, clinical guardrails and MCP with deterministic completions."""

import json
from pathlib import Path

import pytest

from clinical_trial_mvp.agent.analyst import DEFAULT_QUESTION_SQL, PLAN_JSON_SCHEMA, SQL_JSON_SCHEMA
from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.serving.benchmark import Workload, prepare, run_quality
from clinical_trial_mvp.serving.client import Generation, Sample

ROOT = Path(__file__).resolve().parents[2]
PLAN = {"population": "Enrolled or Randomized subjects", "measure": "count subjects",
        "grouping": "treatment arm", "time_window": "none",
        "filters": "screening_status in Enrolled or Randomized", "ranking": "none",
        "needs_clarification": False, "clarification_question": ""}


class FakeClient:
    generation = Generation()

    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def complete(self, messages, schema=None):
        self.calls.append((messages, schema))
        return Sample(success=True, content=json.dumps(next(self.replies)), output_tokens=10)


@pytest.mark.parametrize("repairs", [0, 1])
def test_clinical_repair_and_result_equivalence(repairs):
    workload = Workload(mode="quality", warmups=0, repeats=1, max_repairs=repairs)
    cases, contexts, _, _ = prepare(ROOT, workload, ["eval_001"])
    client = FakeClient([PLAN, {"sql": "...", "rationale": ""},
                         {"sql": DEFAULT_QUESTION_SQL, "rationale": ""}])
    records = []
    summary = run_quality(client, Settings.from_env(), cases, contexts, workload, records.append)
    assert summary["reference_scored"] == 1 and summary["first_pass_correct"] == 0
    assert summary["result_equivalent"] == repairs
    assert len(client.calls) == 2 + repairs
    assert client.calls[0][1] == PLAN_JSON_SCHEMA and client.calls[1][1] == SQL_JSON_SCHEMA
    assert records[0]["scores"]["repair_count"] == repairs
    assert records[0]["attempts"][0]["model_metrics"]["output_tokens"] == 10


def test_policy_cases_are_scored_without_inference_or_sql():
    workload = Workload(mode="quality", warmups=0, repeats=1)
    cases, contexts, _, _ = prepare(ROOT, workload, ["eval_012", "eval_015", "eval_016", "eval_017"])
    client = FakeClient([])
    records = []
    summary = run_quality(client, Settings.from_env(), cases, contexts, workload, records.append)
    assert summary["behavior_scored"] == summary["behavior_correct"] == 4
    assert summary["reference_scored"] == 0 and not client.calls
    assert all(not r["tool_calls"] and not r["attempts"] for r in records)


def test_engine_failure_does_not_reuse_previous_metrics_or_score_as_correct():
    class Broken(FakeClient):
        def complete(self, messages, schema=None):
            return Sample(error="engine OOM")
    workload = Workload(mode="quality", warmups=0, repeats=1)
    cases, contexts, _, _ = prepare(ROOT, workload, ["eval_001"])
    records = []
    summary = run_quality(Broken([]), Settings.from_env(), cases, contexts, workload, records.append)
    assert summary["result_equivalent"] == 0
    assert "OOM" in records[0]["plan_error"]
