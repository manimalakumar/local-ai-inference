"""Load and validate the evaluation question dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {"id", "category", "question", "expected_behavior", "success_criteria"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("The evaluation dataset must be a nonempty JSON array.")

    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or not REQUIRED_FIELDS.issubset(case):
            raise ValueError(f"Evaluation case is missing required fields: {case!r}")
        if not isinstance(case["id"], str) or case["id"] in seen:
            raise ValueError(f"Duplicate or invalid evaluation ID: {case['id']!r}")
        if not isinstance(case["question"], str) or not case["question"].strip():
            raise ValueError(f"Question is empty for {case['id']}.")
        seen.add(case["id"])
    return cases
