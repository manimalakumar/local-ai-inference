"""Identify questions that need a user choice before SQL generation."""

from __future__ import annotations

import re


def clarification_for(question: str) -> str | None:
    """Ask for a definition when an operational term changes the query meaning."""
    text = question.casefold()
    if re.search(r"\bactive subjects?\b", text):
        return "What should count as an active subject: enrolled, randomized, or another status?"
    if re.search(r"\bon[- ]time\b", text):
        return "Should an on-time visit mean completed on or before its planned date, or within a visit window?"
    if re.search(r"\benrolled\b", text) and not re.search(
        r"\benrolled\s+(?:or|and)\s+randomized\b", text
    ):
        return "Should enrolled include randomized subjects, or only subjects with screening_status = 'Enrolled'?"
    return None
