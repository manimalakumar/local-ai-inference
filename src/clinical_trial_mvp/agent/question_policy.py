"""Choose permitted answer, clarification, or refusal behavior."""

from __future__ import annotations

import re


def preflight_response(question: str) -> tuple[str, str] | None:
    """Handle requests that must not become subject-level or unsupported SQL."""
    text = question.casefold()
    if re.search(r"\b(delete|drop|update|insert)\b", text) and re.search(
        r"\b(subjects?|records?|sites?|trials?|visits?)\b", text
    ):
        return ("refused", "I cannot change or delete records. This database is read-only.")
    if "medical record" in text:
        return (
            "refused",
            "Medical records are not in this dataset, and I cannot display subject-level medical records. I can summarize serious adverse events in aggregate.",
        )
    if re.search(r"\bwhich treatment is better\b|\bclinical efficacy\b", text):
        return (
            "insufficient_data",
            "This synthetic operational dataset cannot determine which treatment is clinically better. That requires an appropriate clinical efficacy analysis.",
        )
    if re.search(r"\bgrade\s*3\b", text):
        return (
            "insufficient_data",
            "Adverse-event severity is recorded as Mild, Moderate, or Severe, without numeric clinical grades. I cannot identify Grade 3 or higher events from this dataset.",
        )
    return None
