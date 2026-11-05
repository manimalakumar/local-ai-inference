"""Check that write and delete attempts are blocked."""

from __future__ import annotations

from typing import Any

from clinical_trial_mvp.data.privacy import QueryPolicyError, validate_readonly_sql


WRITE_PROBES = (
    "DELETE FROM subjects",
    "UPDATE subjects SET screening_status = 'Enrolled'",
    "INSERT INTO subjects (subject_id) VALUES ('SUBJ-9999')",
    "CREATE TABLE unwanted (id INTEGER)",
    "DROP TABLE subjects",
    "ALTER TABLE subjects ADD COLUMN unwanted INTEGER",
    "TRUNCATE TABLE subjects",
    "COPY subjects TO '/tmp/subjects.csv'",
    "SELECT subject_id FROM subjects; DELETE FROM subjects",
    "WITH removed AS (DELETE FROM subjects RETURNING subject_id) SELECT subject_id FROM removed",
)


def score_write_blocking() -> dict[str, Any]:
    results = []
    for sql in WRITE_PROBES:
        try:
            validate_readonly_sql(sql)
        except QueryPolicyError as exc:
            results.append({"sql": sql, "blocked": True, "reason": str(exc)})
        else:
            results.append({"sql": sql, "blocked": False, "reason": None})
    return {
        "blocked": sum(item["blocked"] for item in results),
        "total": len(results),
        "probes": results,
    }
