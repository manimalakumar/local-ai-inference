"""Freeze reviewed, per-question model errors alongside raw evaluation records."""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from clinical_trial_mvp.data.schema import TABLE_ORDER, TABLE_SCHEMAS


FAILURE_CAUSES = {
    "missing context", "invalid schema reference", "wrong join", "wrong population",
    "wrong measure", "wrong grouping/ranking", "wrong time window/filter", "ambiguous question",
}


def freeze_error_register(
    detail_path: Path, review_path: Path, output_path: Path, *, code_zip_path: Path | None = None
) -> dict[str, Any]:
    records = [json.loads(line) for line in detail_path.read_text(encoding="utf-8").splitlines()]
    reviews = json.loads(review_path.read_text(encoding="utf-8"))
    if set(reviews) != {record["id"] for record in records}:
        raise ValueError("Review IDs must match every detail record exactly")
    items = []
    for record in records:
        review = reviews[record["id"]]
        cause = review.get("primary_failure_cause")
        if cause is not None and cause not in FAILURE_CAUSES:
            raise ValueError(f"Invalid primary failure cause for {record['id']}: {cause}")
        if not isinstance(review.get("secondary_tags", []), list):
            raise ValueError(f"secondary_tags must be a list: {record['id']}")
        if review.get("reviewed_semantic_correct") not in (True, False, None):
            raise ValueError(f"Invalid reviewed semantic score: {record['id']}")
        attempts = record["attempts"]
        final = attempts[-1]
        items.append({
            "id": record["id"],
            "category": record["category"],
            "question": record["question"],
            "expected_behavior": record["expected_behavior"],
            "schema_mode": "selected" if record.get("schema_selection") else "full",
            "selected_tables": record.get("schema_selection", {}).get("tables", TABLE_ORDER),
            "selected_columns": record.get("schema_selection", {}).get(
                "columns", {table: [column.name for column in TABLE_SCHEMAS[table]] for table in TABLE_ORDER}
            ),
            "generated_sql": final.get("sql"),
            "first_sql": attempts[0].get("sql"),
            "execution_success": record["scores"]["sql_execution_success"],
            "execution_error": final.get("error"),
            "row_match": record["scores"]["result_equivalence"],
            "reviewed_semantic_correct": review["reviewed_semantic_correct"],
            "primary_failure_cause": cause,
            "secondary_tags": review.get("secondary_tags", []),
            "review_note": review.get("review_note", ""),
            "repair_count": record["scores"]["repair_count"],
            "plan": record.get("plan"),
            "plan_error": record.get("plan_error"),
        })
    referenced = [item for item in items if item["row_match"] is not None]
    payload = {
        "register_version": 1,
        "run_id": detail_path.stem,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_detail": str(detail_path),
        "review_source": str(review_path),
        "code_zip": str(code_zip_path) if code_zip_path else None,
        "counts": {
            "questions": len(items),
            "reference_cases": len(referenced),
            "row_matches": sum(item["row_match"] is True for item in referenced),
            "reviewed_semantic_correct": sum(item["reviewed_semantic_correct"] is True for item in referenced),
            "unreviewed_reference_cases": sum(item["reviewed_semantic_correct"] is None for item in referenced),
        },
        "questions": items,
    }
    with output_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, indent=2, default=str) + "\n")
    checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    with output_path.with_suffix(".json.sha256").open("x", encoding="utf-8") as sidecar:
        sidecar.write(json.dumps({"file": output_path.name, "sha256": checksum}) + "\n")
    return payload
