"""Run evaluation questions through the actual model and repair path."""

from __future__ import annotations

import json
import signal
import threading
import time
from typing import Any, Callable

from clinical_trial_mvp.agent.analyst import (
    ClinicalTrialAnalystAgent,
    build_plan_prompt,
    build_sql_generation_prompt,
    build_sql_repair_prompt,
)
from clinical_trial_mvp.mcp.client import execute_readonly_sql_via_mcp, get_schema_via_mcp
from clinical_trial_mvp.data.privacy import validate_readonly_sql
from clinical_trial_mvp.agent.question_policy import preflight_response
from clinical_trial_mvp.agent.plan_policy import check_sql_question_filters, finalize_plan
from clinical_trial_mvp.agent.intent_contracts import check_plan_intent, check_sql_intent


class ModelPathRunner:
    """Exercise model generation directly, including the built-in question."""

    def __init__(
        self,
        agent: ClinicalTrialAnalystAgent,
        *,
        sql_executor: Callable[[str], str] | None = None,
        question_timeout_seconds: float = 300.0,
    ) -> None:
        self.agent = agent
        if question_timeout_seconds <= 0:
            raise ValueError("question_timeout_seconds must be positive")
        self.question_timeout_seconds = question_timeout_seconds
        self.sql_executor = sql_executor or (
            lambda sql: execute_readonly_sql_via_mcp(sql, agent.settings)
        )

    def _invoke_with_deadline(self, prompt: str, deadline: float, *, plan_task: bool = False) -> str:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError(f"Question exceeded {self.question_timeout_seconds:g}s evaluation timeout.")
        if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
            return self.agent._invoke_llm(prompt, plan_task=plan_task)
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)

        def expire(_signum, _frame):
            raise TimeoutError(f"Question exceeded {self.question_timeout_seconds:g}s evaluation timeout.")

        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            return self.agent._invoke_llm(prompt, plan_task=plan_task)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0:
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)

    def _attempt(self, prompt: str, question: str, deadline: float) -> dict[str, Any]:
        attempt: dict[str, Any] = {
            "raw_model_response": None,
            "sql": None,
            "sql_safety": None,
            "semantic_filter_check": None,
            "intent_contract_check": None,
            "intent_contract_violations": [],
            "timed_out": False,
            "schema_compliance": None,
            "execution_success": False,
            "query_result": None,
            "error": None,
            "inference_seconds": 0.0,
            "query_seconds": 0.0,
            "privacy_check_seconds": 0.0,
            "model_metrics": {},
        }
        started = time.perf_counter()
        try:
            raw = self._invoke_with_deadline(prompt, deadline)
        except Exception as exc:
            attempt["error"] = f"Model invocation failed: {exc}"
            attempt["timed_out"] = isinstance(exc, TimeoutError)
            return attempt
        finally:
            attempt["inference_seconds"] = time.perf_counter() - started

        attempt["raw_model_response"] = raw
        attempt["model_metrics"] = dict(self.agent.last_invocation_metrics)
        try:
            payload = self.agent._parse_sql_json(raw)
        except (ValueError, KeyError) as exc:
            attempt["error"] = f"Could not parse model SQL: {exc}"
            return attempt
        sql = payload["sql"]
        attempt["sql"] = sql

        privacy_started = time.perf_counter()
        try:
            normalized = validate_readonly_sql(sql)
        except ValueError as exc:
            attempt["sql_safety"] = False
            attempt["error"] = str(exc)
            return attempt
        finally:
            attempt["privacy_check_seconds"] += time.perf_counter() - privacy_started
        attempt["sql_safety"] = True
        try:
            check_sql_question_filters(question, normalized)
        except ValueError as exc:
            attempt["semantic_filter_check"] = False
            attempt["error"] = str(exc)
            return attempt
        attempt["semantic_filter_check"] = True
        violations = check_sql_intent(question, normalized)
        attempt["intent_contract_violations"] = violations
        attempt["intent_contract_check"] = not violations
        if violations:
            attempt["error"] = "Question-intent contract violation: " + " | ".join(violations)
            return attempt

        # EXPLAIN binds table and column names without fetching data.
        try:
            import duckdb

            with duckdb.connect(str(self.agent.settings.db_path), read_only=True) as connection:
                connection.execute(f"EXPLAIN {normalized}")
        except Exception as exc:
            attempt["schema_compliance"] = False
            attempt["error"] = f"DuckDB could not bind or plan the SQL: {exc}"
            return attempt
        attempt["schema_compliance"] = True

        query_started = time.perf_counter()
        try:
            result = json.loads(self.sql_executor(sql))
            if not isinstance(result, dict):
                raise ValueError("SQL tool did not return a JSON object.")
            attempt["query_result"] = result
            attempt["execution_success"] = result.get("ok") is True
            if not attempt["execution_success"]:
                attempt["error"] = result.get("error") or "SQL execution failed."
        except Exception as exc:
            attempt["error"] = f"SQL tool failed: {exc}"
        finally:
            attempt["query_seconds"] = time.perf_counter() - query_started
        return attempt

    def run_case(self, case: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        deadline = started + self.question_timeout_seconds
        question = case["question"]
        policy_started = time.perf_counter()
        policy = preflight_response(question)
        policy_seconds = time.perf_counter() - policy_started
        if policy:
            return {
                "attempts": [], "plan": None, "plan_checks": None,
                "plan_contract_findings": [], "question_timeout_seconds": self.question_timeout_seconds,
                "timed_out": False, "plan_raw_model_response": None,
                "plan_error": None, "plan_prompt": None, "plan_inference_seconds": 0.0,
                "generation_prompt": None, "repair_prompt": None,
                "schema_selection": None, "policy_response": {"status": policy[0], "answer": policy[1]},
                "tool_calls": [], "latency_seconds": time.perf_counter() - started,
                "inference_seconds": 0.0, "query_seconds": 0.0,
                "privacy_check_seconds": policy_seconds,
            }
        schema, selection, guidance = self.agent.schema_context(question)
        plan_prompt = build_plan_prompt(question, schema, guidance)
        plan_started = time.perf_counter()
        plan_raw = None
        plan = None
        plan_checks = None
        plan_contract_findings: list[str] = []
        plan_error = None
        try:
            plan_raw = self._invoke_with_deadline(plan_prompt, deadline, plan_task=True)
            plan, plan_checks = finalize_plan(
                question, self.agent._parse_plan_json(plan_raw, question=question)
            )
            plan_contract_findings = check_plan_intent(question, plan)
        except Exception as exc:
            plan_error = f"Could not generate an answer plan: {exc}"
        plan_inference_seconds = time.perf_counter() - plan_started
        plan_model_metrics = dict(self.agent.last_invocation_metrics) if plan_raw is not None else {}
        generation_prompt = build_sql_generation_prompt(question, schema, plan, plan_contract_findings, guidance) if plan and not plan["needs_clarification"] else None
        if generation_prompt is None:
            attempts = [{
                "raw_model_response": None, "sql": None, "sql_safety": None,
                "schema_compliance": None, "execution_success": False,
                "query_result": None,
                "error": plan_error or (plan["clarification_question"] if plan else "Planning failed."),
                "inference_seconds": 0.0, "query_seconds": 0.0,
            }]
        else:
            attempts = [self._attempt(generation_prompt, question, deadline)]
        first = attempts[0]
        repair_prompt = None
        expects_sql = case["expected_behavior"] in {"answer", "qualified_answer"}
        if self.agent.settings.max_repairs and generation_prompt is not None and expects_sql and not first["execution_success"] and first["raw_model_response"] is not None:
            failed_sql = first["sql"] or first["raw_model_response"]
            repair_prompt = build_sql_repair_prompt(
                question, failed_sql, first["error"] or "SQL generation failed.", schema, plan,
                guidance=guidance,
            )
            attempts.append(self._attempt(repair_prompt, question, deadline))
        tool_calls = ["retrieve_schema" if selection["mode"] == "retrieved" else "get_schema"]
        if plan and plan.get("needs_clarification"):
            tool_calls.append("request_clarification")
        tool_calls.extend("execute_readonly_sql" for attempt in attempts if attempt.get("query_result") is not None)
        return {
            "attempts": attempts,
            "plan": plan,
            "plan_checks": plan_checks,
            "plan_contract_findings": plan_contract_findings,
            "question_timeout_seconds": self.question_timeout_seconds,
            "timed_out": "timeout" in (plan_error or "").casefold() or any(a.get("timed_out") for a in attempts),
            "plan_raw_model_response": plan_raw,
            "plan_error": plan_error,
            "plan_prompt": plan_prompt,
            "plan_inference_seconds": plan_inference_seconds,
            "plan_model_metrics": plan_model_metrics,
            "generation_prompt": generation_prompt,
            "repair_prompt": repair_prompt,
            "schema_selection": selection,
            "policy_response": None,
            "tool_calls": tool_calls,
            "latency_seconds": time.perf_counter() - started,
            "inference_seconds": plan_inference_seconds + sum(a["inference_seconds"] for a in attempts),
            "query_seconds": sum(a["query_seconds"] for a in attempts),
            "privacy_check_seconds": policy_seconds + sum(a.get("privacy_check_seconds", 0) for a in attempts),
            "model_cold_start_seconds": self.agent.model_cold_start_seconds,
        }
