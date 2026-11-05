"""Describe the clinical trial tables, columns, and relationships."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    dtype: str
    description: str


TABLE_SCHEMAS: dict[str, list[Column]] = {
    "trials": [
        Column("trial_id", "VARCHAR", "Primary trial identifier."),
        Column("phase", "VARCHAR", "Clinical phase, such as Phase I through Phase IV."),
        Column("indication", "VARCHAR", "Disease or therapeutic area being studied."),
        Column("start_date", "DATE", "Trial start date."),
        Column("target_enrollment", "INTEGER", "Planned total enrollment for the trial."),
    ],
    "sites": [
        Column("site_id", "VARCHAR", "Primary site identifier."),
        Column("trial_id", "VARCHAR", "Foreign key to trials.trial_id."),
        Column("region", "VARCHAR", "Geographic region for the site."),
        Column("activation_date", "DATE", "Date the site was activated."),
        Column("enrollment_target", "INTEGER", "Planned enrollment for the site."),
    ],
    "subjects": [
        Column("subject_id", "VARCHAR", "Primary synthetic subject identifier. Treat as private."),
        Column("site_id", "VARCHAR", "Foreign key to sites.site_id."),
        Column("treatment_arm", "VARCHAR", "Assigned treatment arm or Not Assigned."),
        Column("screening_status", "VARCHAR", "Screening/enrollment status: Enrolled, Randomized, Screen Failed, or Withdrawn."),
        Column("enrollment_date", "DATE", "Subject enrollment date when available."),
    ],
    "visits": [
        Column("visit_id", "VARCHAR", "Primary visit identifier."),
        Column("subject_id", "VARCHAR", "Foreign key to subjects.subject_id. Treat as private."),
        Column("visit_name", "VARCHAR", "Visit label, such as Screening, Baseline, Week 4."),
        Column("planned_date", "DATE", "Planned visit date."),
        Column("actual_date", "DATE", "Actual visit date when completed."),
        Column("status", "VARCHAR", "Visit status: Completed, Missed, Pending, or Screen Failed."),
    ],
    "adverse_events": [
        Column("subject_id", "VARCHAR", "Foreign key to subjects.subject_id. Treat as private."),
        Column("event_term", "VARCHAR", "Adverse event term."),
        Column("severity", "VARCHAR", "Mild, Moderate, or Severe."),
        Column("serious", "BOOLEAN", "Whether the adverse event is serious."),
        Column("onset_date", "DATE", "Adverse event onset date."),
        Column("outcome", "VARCHAR", "Adverse event outcome."),
    ],
    "screen_failures": [
        Column("screen_failure_id", "VARCHAR", "Primary screen-failure record identifier."),
        Column("subject_id", "VARCHAR", "Foreign key to a Screen Failed subject. Treat as private."),
        Column("site_id", "VARCHAR", "Foreign key to the subject's site."),
        Column("screening_date", "DATE", "Screening date after site activation."),
        Column("failure_category", "VARCHAR", "Inclusion, Exclusion, Lab, Consent, or Withdrawal."),
        Column("criterion_code", "VARCHAR", "Controlled protocol criterion code."),
        Column("failure_reason", "VARCHAR", "Controlled clinical reason for screen failure."),
        Column("rescreen_allowed", "BOOLEAN", "Whether protocol-guided rescreening is allowed."),
        Column("notes", "VARCHAR", "Optional synthetic note."),
    ],
    "protocol_deviations": [
        Column("deviation_id", "VARCHAR", "Primary protocol-deviation identifier."),
        Column("subject_id", "VARCHAR", "Optional foreign key to subjects.subject_id. Treat as private."),
        Column("site_id", "VARCHAR", "Foreign key to sites.site_id."),
        Column("deviation_date", "DATE", "Date the deviation occurred."),
        Column("detected_date", "DATE", "Date detected; same day or later than deviation_date."),
        Column("category", "VARCHAR", "Visit Window, Consent, Dosing, Eligibility, or Procedure."),
        Column("severity", "VARCHAR", "Minor, Major, or Critical."),
        Column("description", "VARCHAR", "Controlled template-based deviation description."),
        Column("reported_to_irb", "BOOLEAN", "Whether the deviation was reported to the IRB."),
        Column("corrective_action", "VARCHAR", "Controlled corrective action."),
        Column("status", "VARCHAR", "Open, Investigating, or Resolved."),
        Column("resolved_date", "DATE", "Resolution date for resolved deviations only."),
    ],
    "lab_results": [
        Column("lab_result_id", "VARCHAR", "Primary laboratory-result identifier."),
        Column("subject_id", "VARCHAR", "Foreign key to an enrolled subject. Treat as private."),
        Column("visit_id", "VARCHAR", "Foreign key to visits.visit_id."),
        Column("collection_date", "DATE", "Collection date within two days of the actual visit."),
        Column("test_code", "VARCHAR", "ALT, AST, HGB, WBC, or CREAT."),
        Column("test_name", "VARCHAR", "Controlled test name mapped from test_code."),
        Column("result_value", "DOUBLE", "Synthetic numeric result sampled around the test range."),
        Column("unit", "VARCHAR", "Controlled unit mapped from test_code."),
        Column("reference_low", "DOUBLE", "Test-specific synthetic reference lower bound."),
        Column("reference_high", "DOUBLE", "Test-specific synthetic reference upper bound."),
        Column("abnormal_flag", "VARCHAR", "Low, Normal, or High, derived from the stored range."),
        Column("clinically_significant", "BOOLEAN", "Rare synthetic clinical-significance flag."),
        Column("reviewed", "BOOLEAN", "Whether the result was reviewed."),
    ],
    "data_queries": [
        Column("query_id", "VARCHAR", "Primary data-query identifier."),
        Column("subject_id", "VARCHAR", "Foreign key to subjects.subject_id. Treat as private."),
        Column("site_id", "VARCHAR", "Foreign key to the subject's site."),
        Column("visit_id", "VARCHAR", "Optional foreign key to visits.visit_id."),
        Column("opened_date", "DATE", "Date opened, on or after the source-data date."),
        Column("query_type", "VARCHAR", "Missing, Inconsistent, Outlier, or Clarification."),
        Column("field_name", "VARCHAR", "Controlled queried field name."),
        Column("priority", "VARCHAR", "Low, Medium, or High."),
        Column("query_text", "VARCHAR", "Controlled template-based clinical query."),
        Column("status", "VARCHAR", "Open, Answered, or Closed."),
        Column("response_date", "DATE", "Response date for answered or closed queries."),
        Column("closed_date", "DATE", "Closure date for closed queries."),
        Column("resolution_hours", "INTEGER", "Whole hours from opened_date to closed_date."),
    ],
    "site_monitoring": [
        Column("monitoring_visit_id", "VARCHAR", "Primary monitoring-visit identifier."),
        Column("site_id", "VARCHAR", "Foreign key to sites.site_id."),
        Column("visit_date", "DATE", "Monitoring date after site activation."),
        Column("visit_type", "VARCHAR", "Initiation, Interim, For-cause, or Closeout."),
        Column("monitor_id", "VARCHAR", "Synthetic monitor identifier."),
        Column("risk_rating", "VARCHAR", "Low, Medium, or High."),
        Column("findings_count", "INTEGER", "Total findings, usually zero through eight."),
        Column("critical_findings", "INTEGER", "Critical findings; no more than findings_count."),
        Column("open_actions", "INTEGER", "Open actions; no more than findings_count."),
        Column("report_date", "DATE", "Report date one through ten days after visit_date."),
        Column("next_visit_date", "DATE", "Optional next visit date 30 through 120 days later."),
        Column("status", "VARCHAR", "Planned, Completed, or Report Pending."),
        Column("notes", "VARCHAR", "Controlled summary with optional synthetic detail."),
    ],
}


TABLE_ORDER = tuple(TABLE_SCHEMAS)


RELATIONSHIPS = [
    "sites.trial_id -> trials.trial_id",
    "subjects.site_id -> sites.site_id",
    "visits.subject_id -> subjects.subject_id",
    "adverse_events.subject_id -> subjects.subject_id",
    "screen_failures.subject_id -> subjects.subject_id",
    "screen_failures.site_id -> sites.site_id",
    "protocol_deviations.subject_id -> subjects.subject_id (nullable)",
    "protocol_deviations.site_id -> sites.site_id",
    "lab_results.subject_id -> subjects.subject_id",
    "lab_results.visit_id -> visits.visit_id",
    "data_queries.subject_id -> subjects.subject_id",
    "data_queries.site_id -> sites.site_id",
    "data_queries.visit_id -> visits.visit_id (nullable)",
    "site_monitoring.site_id -> sites.site_id",
]


def schema_prompt() -> str:
    lines = ["DuckDB clinical trial schema:"]
    for table_name, columns in TABLE_SCHEMAS.items():
        lines.append(f"\nTable: {table_name}")
        for column in columns:
            lines.append(f"- {column.name} {column.dtype}: {column.description}")

    lines.append("\nRelationships:")
    for relationship in RELATIONSHIPS:
        lines.append(f"- {relationship}")

    lines.append(
        "\nPrivacy note: subject_id is a subject-level identifier. Prefer aggregate results. "
        "If subject_id appears in query results, the MCP server pseudonymizes it before display."
    )
    return "\n".join(lines)


SCHEMA_PROMPT = schema_prompt()


CREATE_TABLE_SQL = """
CREATE TABLE trials (
    trial_id VARCHAR PRIMARY KEY,
    phase VARCHAR,
    indication VARCHAR,
    start_date DATE,
    target_enrollment INTEGER
);

CREATE TABLE sites (
    site_id VARCHAR PRIMARY KEY,
    trial_id VARCHAR REFERENCES trials(trial_id),
    region VARCHAR,
    activation_date DATE,
    enrollment_target INTEGER
);

CREATE TABLE subjects (
    subject_id VARCHAR PRIMARY KEY,
    site_id VARCHAR REFERENCES sites(site_id),
    treatment_arm VARCHAR,
    screening_status VARCHAR,
    enrollment_date DATE
);

CREATE TABLE visits (
    visit_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR REFERENCES subjects(subject_id),
    visit_name VARCHAR,
    planned_date DATE,
    actual_date DATE,
    status VARCHAR
);

CREATE TABLE adverse_events (
    subject_id VARCHAR REFERENCES subjects(subject_id),
    event_term VARCHAR,
    severity VARCHAR,
    serious BOOLEAN,
    onset_date DATE,
    outcome VARCHAR
);

CREATE TABLE screen_failures (
    screen_failure_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR REFERENCES subjects(subject_id),
    site_id VARCHAR REFERENCES sites(site_id),
    screening_date DATE,
    failure_category VARCHAR,
    criterion_code VARCHAR,
    failure_reason VARCHAR,
    rescreen_allowed BOOLEAN,
    notes VARCHAR
);

CREATE TABLE protocol_deviations (
    deviation_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR REFERENCES subjects(subject_id),
    site_id VARCHAR REFERENCES sites(site_id),
    deviation_date DATE,
    detected_date DATE,
    category VARCHAR,
    severity VARCHAR,
    description VARCHAR,
    reported_to_irb BOOLEAN,
    corrective_action VARCHAR,
    status VARCHAR,
    resolved_date DATE
);

CREATE TABLE lab_results (
    lab_result_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR REFERENCES subjects(subject_id),
    visit_id VARCHAR REFERENCES visits(visit_id),
    collection_date DATE,
    test_code VARCHAR,
    test_name VARCHAR,
    result_value DOUBLE,
    unit VARCHAR,
    reference_low DOUBLE,
    reference_high DOUBLE,
    abnormal_flag VARCHAR,
    clinically_significant BOOLEAN,
    reviewed BOOLEAN
);

CREATE TABLE data_queries (
    query_id VARCHAR PRIMARY KEY,
    subject_id VARCHAR REFERENCES subjects(subject_id),
    site_id VARCHAR REFERENCES sites(site_id),
    visit_id VARCHAR REFERENCES visits(visit_id),
    opened_date DATE,
    query_type VARCHAR,
    field_name VARCHAR,
    priority VARCHAR,
    query_text VARCHAR,
    status VARCHAR,
    response_date DATE,
    closed_date DATE,
    resolution_hours INTEGER
);

CREATE TABLE site_monitoring (
    monitoring_visit_id VARCHAR PRIMARY KEY,
    site_id VARCHAR REFERENCES sites(site_id),
    visit_date DATE,
    visit_type VARCHAR,
    monitor_id VARCHAR,
    risk_rating VARCHAR,
    findings_count INTEGER,
    critical_findings INTEGER,
    open_actions INTEGER,
    report_date DATE,
    next_visit_date DATE,
    status VARCHAR,
    notes VARCHAR
);
""".strip()
