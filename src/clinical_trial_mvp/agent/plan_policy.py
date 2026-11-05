"""Check model plans for required fields and obvious filter errors."""

from __future__ import annotations

import re
from typing import Any


PLAN_FIELDS = ("population", "measure", "grouping", "time_window", "filters")


def _missing_metric(question: str) -> bool:
    """A causal performance question needs a metric before it can be queried."""
    return bool(
        re.search(r"\bwhy\b.*\bperform(?:ing)? poorly\b", question, re.I)
        and not re.search(
            r"\b(enrollment|screen.fail|missed visits?|data quer(?:y|ies)|"
            r"protocol deviations?|adverse events?|resolution time)\b", question, re.I
        )
    )


def finalize_plan(question: str, plan: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep material clarifications and make model-only ambiguity non-blocking."""
    adjusted = dict(plan)
    missing = [key for key in PLAN_FIELDS if key != "filters" and not adjusted[key].strip()]
    if missing:
        raise ValueError(f"Plan has empty required fields: {', '.join(missing)}")
    adjusted["filters"] = adjusted["filters"].strip() or "none"

    model_requested = adjusted["needs_clarification"]
    material = _missing_metric(question)
    adjusted["needs_clarification"] = material
    if material:
        adjusted["clarification_question"] = (
            "Which measure should I use to assess performance (for example enrollment, "
            "missed visits, or open queries)? I cannot determine a cause from this question alone."
        )
    else:
        adjusted["clarification_question"] = ""

    serious_only = bool(re.search(r"\bserious\b", question, re.I)) and not bool(
        re.search(r"\bsevere\b|\bgrade\s*\d", question, re.I)
    )
    removed_severe = False
    if serious_only:
        for key in PLAN_FIELDS + ("ranking",):
            value = adjusted[key]
            cleaned = re.sub(
                r"(?:\b(?:\w+\.)?severity\s*(?:=|==|IN\s*\()\s*['\"]?Severe['\"]?\s*\)?)",
                "",
                value,
                flags=re.I,
            )
            if cleaned != value:
                adjusted[key] = cleaned.strip(" ,;and") or "none"
                removed_severe = True
    return adjusted, {
        "model_requested_clarification": model_requested,
        "material_ambiguity": material,
        "clarification_suppressed": bool(model_requested and not material),
        "removed_unrequested_severe_from_plan": removed_severe,
    }


def check_sql_question_filters(question: str, sql: str) -> None:
    """Reject a Severe restriction when the request is for serious events alone."""
    if not re.search(r"\bserious\b", question, re.I) or re.search(
        r"\bsevere\b|\bgrade\s*\d", question, re.I
    ):
        return
    # Strip comments; retain literals because the comparison value is the issue.
    code = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.S)
    severity_predicate = (
        r"\b(?:\w+\.)?severity\s*(?:=|==|IN\s*\()\s*['\"]Severe['\"]"
        r"|['\"]Severe['\"]\s*=\s*(?:\w+\.)?severity\b"
    )
    if re.search(severity_predicate, code, re.I):
        raise ValueError(
            "Unrequested severity='Severe' filter: serious adverse events must use serious=true "
            "without restricting severity."
        )
