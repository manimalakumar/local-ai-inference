"""Deterministic, question-specific schema excerpt for the SQL prompt."""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from clinical_trial_mvp.data.schema import RELATIONSHIPS, TABLE_ORDER, TABLE_SCHEMAS


SCHEMA_SELECTION_VERSION = "v1-relationship-closure"

# These cues identify tables, not answers. The glossary and SQL rules remain separate.
TABLE_CUES = {
    "trials": r"\btrials?\b|\bclinical phase\b|\bindication\b",
    "sites": r"\bsites?\b|\bregions?\b|\bactivation\b",
    "subjects": r"\bsubjects?\b|\bpatients?\b|\benroll(?:ed|ment|ing)?\b|\brandomiz\w*\b|\btreatment arms?\b|\bscreening status\b",
    "visits": r"\bvisits?\b|\bmissed\b|\bon.time\b|\bdelayed\b",
    "adverse_events": r"\badverse.event\w*\b|\bserious event\w*\b|\bsae\b",
    "screen_failures": r"\bscreen.fail\w*\b|\bscreening fail\w*\b",
    "protocol_deviations": r"\bprotocol.deviation\w*\b|\bdeviation rates?\b",
    "lab_results": r"\blab(?:oratory)?\b|\balt\b|\bast\b|\bhgb\b|\bwbc\b|\bcreat\b",
    "data_queries": r"\bdata.quer\w*\b|\bopen quer\w*\b|\bquer\w* resolution\b",
    "site_monitoring": r"\bmonitor\w*\b",
}

# A small set of generally useful measure columns is retained for selected tables.
# Question-matched columns and all relationship keys are added below.
CORE_COLUMNS = {
    "trials": {"trial_id", "phase", "target_enrollment"},
    "sites": {"site_id", "trial_id", "region", "activation_date", "enrollment_target"},
    "subjects": {"subject_id", "site_id", "treatment_arm", "screening_status", "enrollment_date"},
    "visits": {"visit_id", "subject_id", "visit_name", "planned_date", "actual_date", "status"},
    "adverse_events": {"subject_id", "severity", "serious", "onset_date"},
    "screen_failures": {"screen_failure_id", "subject_id", "site_id", "screening_date"},
    "protocol_deviations": {"deviation_id", "subject_id", "site_id", "deviation_date", "status"},
    "lab_results": {"lab_result_id", "subject_id", "visit_id", "test_code", "result_value", "abnormal_flag"},
    "data_queries": {"query_id", "subject_id", "site_id", "visit_id", "status", "resolution_hours", "opened_date", "closed_date"},
    "site_monitoring": {"monitoring_visit_id", "site_id", "visit_date", "status"},
}

COLUMN_CUES = {
    "failure_category": r"\b(?:failure )?categor\w*\b",
    "criterion_code": r"\bcriteri\w*\b",
    "failure_reason": r"\breason\b",
    "rescreen_allowed": r"\brescreen\w*\b",
    "category": r"\bcategor\w*\b",
    "severity": r"\b(?:severity|severe|grade)\b",
    "detected_date": r"\bdetect\w*\b",
    "resolved_date": r"\bresolve\w*\b",
    "reported_to_irb": r"\birb\b",
    "corrective_action": r"\bcorrective\b|\baction\w*\b",
    "clinically_significant": r"\bclinically significant\b",
    "reference_low": r"\breference (?:low|range)\b",
    "reference_high": r"\breference (?:high|range)\b",
    "reviewed": r"\breview\w*\b",
    "query_type": r"\bquery type\b",
    "field_name": r"\bfield\b",
    "priority": r"\bpriorit\w*\b",
    "response_date": r"\bresponse\b|\banswer\w*\b",
    "visit_type": r"\bvisit type\b|\bmonitoring type\b",
    "risk_rating": r"\brisk\b",
    "findings_count": r"\bfindings?\b",
    "critical_findings": r"\bcritical findings?\b",
    "open_actions": r"\bopen actions?\b",
    "report_date": r"\breport\w*\b",
    "next_visit_date": r"\bnext visit\b",
}


def _relationship_parts(relationship: str) -> tuple[str, str, str, str]:
    match = re.match(r"(\w+)\.(\w+) -> (\w+)\.(\w+)", relationship)
    if not match:
        raise ValueError(f"Invalid relationship: {relationship}")
    return match.groups()


def _shortest_path(start: str, goal: str) -> list[str]:
    graph: dict[str, list[str]] = {table: [] for table in TABLE_ORDER}
    for relationship in RELATIONSHIPS:
        left, _, right, _ = _relationship_parts(relationship)
        graph[left].append(right)
        graph[right].append(left)
    paths = deque([[start]])
    visited = {start}
    while paths:
        path = paths.popleft()
        if path[-1] == goal:
            return path
        for neighbor in graph[path[-1]]:
            if neighbor not in visited:
                visited.add(neighbor)
                paths.append([*path, neighbor])
    raise ValueError(f"No relationship path from {start} to {goal}")


@dataclass(frozen=True)
class SchemaSelection:
    tables: tuple[str, ...]
    columns: dict[str, tuple[str, ...]]
    prompt: str


def select_schema(question: str, full_schema: str) -> SchemaSelection:
    """Select question tables and columns, closing indirect relationship paths."""
    # A monitoring visit belongs to site_monitoring, not the protocol visits table.
    protocol_question = re.sub(r"\bmonitoring visits?\b", "", question, flags=re.I)
    seeds = [
        table for table in TABLE_ORDER
        if re.search(TABLE_CUES[table], protocol_question if table == "visits" else question, re.I)
    ]
    if not seeds:
        # Unknown terminology must not silently hide a possibly useful table.
        return SchemaSelection(
            tables=TABLE_ORDER,
            columns={t: tuple(c.name for c in TABLE_SCHEMAS[t]) for t in TABLE_ORDER},
            prompt=full_schema,
        )

    selected = set(seeds)
    for index, start in enumerate(seeds):
        for goal in seeds[index + 1:]:
            selected.update(_shortest_path(start, goal))

    # A question may mention a direct site key and a visit. Retain the subject
    # bridge even when another selected table also has a direct site key.
    if "visits" in selected and "sites" in selected:
        selected.add("subjects")
    if "adverse_events" in selected and "sites" in selected:
        selected.add("subjects")

    tables = tuple(table for table in TABLE_ORDER if table in selected)
    keys = {table: set() for table in tables}
    relationships = []
    for relationship in RELATIONSHIPS:
        left, left_key, right, right_key = _relationship_parts(relationship)
        if left in selected and right in selected:
            relationships.append(relationship)
            keys[left].add(left_key)
            keys[right].add(right_key)

    lines = ["DuckDB clinical trial schema:", "Selected tables and columns for this question:"]
    columns: dict[str, tuple[str, ...]] = {}
    for table in tables:
        allowed = CORE_COLUMNS[table] | keys[table]
        for column in TABLE_SCHEMAS[table]:
            if re.search(rf"\b{re.escape(column.name.replace('_', ' '))}\b", question, re.I):
                allowed.add(column.name)
            cue = COLUMN_CUES.get(column.name)
            if cue and re.search(cue, question, re.I):
                allowed.add(column.name)
        chosen = tuple(column.name for column in TABLE_SCHEMAS[table] if column.name in allowed)
        columns[table] = chosen
        lines.append(f"\nTable: {table}")
        for column in TABLE_SCHEMAS[table]:
            if column.name in allowed:
                lines.append(f"- {column.name} {column.dtype}: {column.description}")

    lines.append("\nRelationships between selected tables:")
    lines.extend(f"- {relationship}" for relationship in relationships)
    lines.append(
        "\nPrivacy note: subject_id is a subject-level identifier. Prefer aggregate results. "
        "If subject_id appears in query results, the MCP server pseudonymizes it before display."
    )
    return SchemaSelection(tables=tables, columns=columns, prompt="\n".join(lines))
