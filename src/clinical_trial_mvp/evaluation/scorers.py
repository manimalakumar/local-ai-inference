"""Deterministically score SQL validity, execution, and answer equivalence."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any


SCORER_VERSION = "mvp1-v1-value-equivalence-and-schema-coverage"


def sql_was_generated(sql: str | None) -> bool:
    value = (sql or "").strip()
    return bool(value) and not bool(re.fullmatch(r"\.{3,}", value))


def _initial_sql_generated(attempt: dict[str, Any]) -> bool:
    if attempt.get("sql") is not None:
        return sql_was_generated(attempt["sql"])
    raw = (attempt.get("raw_model_response") or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:sql)?\s*|\s*```$", "", raw, flags=re.IGNORECASE).strip()
    return bool(re.match(r"^(SELECT|WITH|DELETE|UPDATE|INSERT|CREATE|DROP)\b", raw, re.IGNORECASE))


def _column_name(name: str) -> str:
    return re.sub(r"\s+", "_", name.strip().casefold())


def _number(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        return None
    try:
        return Decimal(str(value))
    except InvalidOperation:
        return None


def _values_equal(actual: Any, expected: Any, abs_tol: Decimal, rel_tol: Decimal) -> bool:
    actual_number = _number(actual)
    expected_number = _number(expected)
    if actual_number is not None and expected_number is not None:
        if not actual_number.is_finite() or not expected_number.is_finite():
            return actual_number == expected_number
        difference = abs(actual_number - expected_number)
        return difference <= max(abs_tol, rel_tol * max(abs(actual_number), abs(expected_number)))
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    return actual == expected


def _normalized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {_column_name(str(key)): value for key, value in row.items()}


def _rows_equal(
    actual: dict[str, Any], expected: dict[str, Any], abs_tol: Decimal, rel_tol: Decimal
) -> bool:
    if actual.keys() != expected.keys():
        return False
    return all(_values_equal(actual[key], expected[key], abs_tol, rel_tol) for key in actual)


def _unique_rows(rows: list[dict[str, Any]], abs_tol: Decimal, rel_tol: Decimal) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    for row in rows:
        if not any(_rows_equal(row, other, abs_tol, rel_tol) for other in unique):
            unique.append(row)
    return unique


def results_equivalent(
    actual: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    order_sensitive: bool = False,
    preserve_duplicates: bool = True,
    require_column_names: bool = False,
    abs_tol: float = 1e-6,
    rel_tol: float = 1e-6,
) -> bool:
    """Compare rows and numeric values, allowing harmless output aliases by default.

    Unordered comparison uses a multiset by default; set mode drops duplicates.
    """
    tolerance_abs = Decimal(str(abs_tol))
    tolerance_rel = Decimal(str(rel_tol))
    actual_rows = [_normalized_row(row) for row in actual]
    expected_rows = [_normalized_row(row) for row in expected]
    if actual_rows and expected_rows:
        actual_columns = list(actual_rows[0])
        expected_columns = list(expected_rows[0])
        if len(actual_columns) != len(expected_columns):
            return False
        if set(actual_columns) != set(expected_columns):
            if require_column_names:
                return False
            shared = set(actual_columns) & set(expected_columns)
            unmatched_actual = [name for name in actual_columns if name not in shared]
            unmatched_expected = [name for name in expected_columns if name not in shared]
            if len(unmatched_actual) != len(unmatched_expected):
                return False
            mapping = {name: name for name in shared}
            mapping.update(zip(unmatched_actual, unmatched_expected))
            actual_rows = [
                {mapping[name]: value for name, value in row.items()} for row in actual_rows
            ]
    if not preserve_duplicates:
        actual_rows = _unique_rows(actual_rows, tolerance_abs, tolerance_rel)
        expected_rows = _unique_rows(expected_rows, tolerance_abs, tolerance_rel)
    if len(actual_rows) != len(expected_rows):
        return False
    if order_sensitive:
        return all(
            _rows_equal(a, e, tolerance_abs, tolerance_rel)
            for a, e in zip(actual_rows, expected_rows)
        )

    # Find a one-to-one match; a greedy match can fail when tolerance ranges overlap.
    matches: dict[int, int] = {}

    def match(actual_index: int, visited: set[int]) -> bool:
        for expected_index, expected_row in enumerate(expected_rows):
            if expected_index in visited or not _rows_equal(
                actual_rows[actual_index], expected_row, tolerance_abs, tolerance_rel
            ):
                continue
            visited.add(expected_index)
            if expected_index not in matches or match(matches[expected_index], visited):
                matches[expected_index] = actual_index
                return True
        return False

    return all(match(index, set()) for index in range(len(actual_rows)))


def score_case(
    case: dict[str, Any],
    attempts: list[dict[str, Any]],
    reference_result: dict[str, Any] | None,
) -> dict[str, Any]:
    expects_sql = case["expected_behavior"] in {"answer", "qualified_answer"}
    first = attempts[0] if attempts else {}
    final = attempts[-1] if attempts else {}
    has_reference = reference_result is not None

    def correct(attempt: dict[str, Any]) -> bool:
        return bool(
            has_reference
            and attempt.get("execution_success")
            and results_equivalent(
                attempt["query_result"]["rows"],
                reference_result["rows"],
                order_sensitive=case.get("order_sensitive", False),
                preserve_duplicates=case.get("preserve_duplicates", True),
                require_column_names=case.get("require_column_names", False),
                abs_tol=case.get("abs_tol", 1e-6),
                rel_tol=case.get("rel_tol", 1e-6),
            )
        )

    repair_count = max(0, len(attempts) - 1)
    final_success = bool(final.get("execution_success"))
    final_correct = correct(final) if has_reference else None
    sql_attempts = [attempt for attempt in attempts if attempt.get("sql")]
    return {
        "sql_generated": _initial_sql_generated(first) if expects_sql else None,
        "sql_safety": all(attempt.get("sql_safety") is True for attempt in sql_attempts) if sql_attempts else None,
        "schema_compliance": final.get("schema_compliance") if final.get("sql_safety") else None,
        "sql_execution_success": final_success if expects_sql else None,
        "result_equivalence": final_correct,
        "first_pass_accuracy": correct(first) if has_reference else None,
        "repair_count": repair_count,
        "repair_success": (
            (final_correct if has_reference else final_success)
            if repair_count and expects_sql else None
        ),
        "schema_linking_accuracy": None,
    }
