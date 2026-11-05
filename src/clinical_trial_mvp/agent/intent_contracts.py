"""Narrow, auditable intent contracts for unambiguous clinical-operations wording.

These checks identify specific errors; they do not prove general SQL correctness.
Plan findings guide generation. SQL findings block execution and consume the
existing single repair opportunity.
"""

from __future__ import annotations

import re
from typing import Any


CONTRACT_VERSION = "v1-zero-date-serious-before-after"


def _kind(question: str) -> str | None:
    q = question.casefold()
    if re.search(r"\benrolled at each site\b|\bsites (?:are )?below 50%", q):
        return "site_enrollment"
    if re.search(r"\bvisits were missed at each site\b", q):
        return "site_missed_visits"
    if re.search(r"\bsites belong to each trial\b", q):
        return "trial_sites"
    if "enrolled-subject count" in q and "for each trial" in q:
        return "trial_enrollment"
    return None


def _serious_kind(question: str) -> str | None:
    serious = bool(re.search(r"\bserious\b", question, re.I))
    severe = bool(re.search(r"\bsevere\b", question, re.I))
    if serious and not severe:
        return "serious"
    if severe and not serious:
        return "severe"
    return None


def _last_month(question: str) -> bool:
    return bool(re.search(r"\b(?:scheduled )?visits?\b.*\bmissed last month\b", question, re.I))


def _before_after(question: str) -> bool:
    return bool(re.search(r"\bprotocol.deviation rates? before and after\b.*\bmonitoring visit\b", question, re.I))


def check_plan_intent(question: str, plan: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    kind = _kind(question)
    plan_text = " ".join(str(plan.get(key, "")) for key in ("population", "measure", "filters")).casefold()
    if kind and not re.search(r"\bzero\b|\ball sites\b|\ball trials\b|\bincluding (?:sites|trials) without\b", plan_text):
        noun = "sites" if kind.startswith("site") else "trials"
        findings.append(f"zero_groups: Include every {noun} group, including zero qualifying rows; start from the parent table with outer joins.")
    if _last_month(question):
        window = str(plan.get("time_window", "")).casefold()
        if "planned_date" not in plan_text + " " + window or not re.search(r"previous calendar month|prior calendar month", window):
            findings.append("previous_calendar_month: Count Missed visits by planned_date in the entire previous calendar month, with inclusive start and exclusive current-month start.")
    serious_kind = _serious_kind(question)
    if serious_kind == "serious" and "severe" in plan_text:
        findings.append("serious_not_severe: Serious events use serious=true regardless of severity; do not describe or filter them as Severe.")
    if serious_kind == "severe" and re.search(r"\bserious\b", plan_text):
        findings.append("severe_not_serious: Severe events use severity='Severe'; do not add an unrequested serious=true restriction.")
    if _before_after(question):
        window = str(plan.get("time_window", "")).casefold()
        measure = str(plan.get("measure", "")).casefold()
        if not ("before" in window + measure and "after" in window + measure and re.search(r"equal|same|90.day", window)):
            findings.append("before_after_windows: Compare deviation counts in equal 90-day windows before and after each monitoring visit; exclude the visit date itself.")
        if re.search(r"\bstatus\b", str(plan.get("filters", "")), re.I):
            findings.append("before_after_population: Include all protocol-deviation statuses unless the question restricts them.")
    return findings


def check_sql_intent(question: str, sql: str) -> list[str]:
    findings: list[str] = []
    code = re.sub(r"/\*.*?\*/|--[^\n]*", " ", sql, flags=re.S)
    compact = re.sub(r"\s+", " ", code).casefold()
    kind = _kind(question)
    if kind:
        parent = "sites" if kind.startswith("site") else "trials"
        children = {
            "site_enrollment": ("subjects",),
            "site_missed_visits": ("subjects", "visits"),
            "trial_sites": ("sites",),
            "trial_enrollment": ("sites", "subjects"),
        }[kind]
        if not re.search(rf"\bfrom\s+{parent}\b", compact) or any(
            not re.search(rf"\bleft\s+(?:outer\s+)?join\s+{child}\b", compact) for child in children
        ):
            findings.append(f"zero_groups: Preserve every {parent} row with LEFT JOIN paths through {', '.join(children)}.")
        where = re.search(r"\bwhere\b(.*?)(?:\bgroup\s+by\b|\bhaving\b|\border\s+by\b|$)", compact)
        if where:
            child_where = (
                r"\b(?:\w+\.)?screening_status\b" if kind in {"site_enrollment", "trial_enrollment"}
                else r"\b(?:v|visits)\.status\b|\bstatus\s*=\s*'missed'" if kind == "site_missed_visits"
                else None
            )
            if child_where and re.search(child_where, where.group(1)):
                findings.append("zero_groups_filter: Put qualifying child filters in JOIN ON or conditional aggregates, not WHERE, so empty groups survive.")
        if re.search(r"\bcount\s*\(\s*\*\s*\)", compact):
            findings.append("zero_groups_count: Count a nullable child key, not COUNT(*), so empty groups return zero.")
    if _last_month(question):
        has_date = "planned_date" in compact and "actual_date" not in compact
        has_bounds = bool(
            re.search(r"planned_date\s*>=\s*date_trunc\s*\(\s*'month'\s*,\s*current_date\s*\)\s*-\s*interval\s*'?1'?\s*month", compact)
            and re.search(r"planned_date\s*<\s*date_trunc\s*\(\s*'month'\s*,\s*current_date\s*\)", compact)
        )
        if not (has_date and has_bounds):
            findings.append("previous_calendar_month: Filter visits.planned_date from date_trunc('month', CURRENT_DATE) - INTERVAL 1 MONTH (inclusive) to date_trunc('month', CURRENT_DATE) (exclusive); do not use actual_date or a rolling cutoff.")
    serious_kind = _serious_kind(question)
    if serious_kind == "serious":
        if not re.search(r"\b(?:\w+\.)?serious\s*(?:=\s*true|is\s+true)\b", compact):
            findings.append("serious_filter: Serious adverse events require serious=true.")
        if re.search(r"\b(?:\w+\.)?severity\s*(?:=|in\s*\(|is\s+not\b)|\bas\s+\w*severe\w*", compact):
            findings.append("serious_not_severe: Do not restrict or label serious adverse events as Severe unless explicitly requested.")
    elif serious_kind == "severe":
        if not re.search(r"\b(?:\w+\.)?severity\s*=\s*'severe'", compact):
            findings.append("severe_filter: Severe adverse events require severity='Severe'.")
        if re.search(r"\b(?:\w+\.)?serious\s*(?:=\s*true|is\s+true)\b", compact):
            findings.append("severe_not_serious: Do not add serious=true to a Severe-only question.")
    if _before_after(question):
        minus = re.findall(r"visit_date\s*-\s*interval\s*'?([0-9]+)'?\s*day", compact)
        plus = re.findall(r"visit_date\s*\+\s*interval\s*'?([0-9]+)'?\s*day", compact)
        before = bool(re.search(r"deviation_date\s*<\s*(?:\w+\.)?visit_date", compact))
        after = bool(re.search(r"deviation_date\s*>\s*(?:\w+\.)?visit_date", compact))
        if not (minus and plus and set(minus) & set(plus) and before and after):
            findings.append("before_after_windows: Use matching day intervals on both sides of monitoring visit_date and separate deviation_date < visit_date from deviation_date > visit_date.")
        if re.search(r"\b(?:pd|protocol_deviations)\.status\s*(?:=|in\s*\()", compact):
            findings.append("before_after_population: Include all protocol-deviation statuses unless requested otherwise.")
    return findings
