"""Keep inherited baseline tests on the full-schema mode they originally exercised."""

import pytest


@pytest.fixture(autouse=True)
def baseline_full_schema(monkeypatch):
    monkeypatch.setenv("SCHEMA_CONTEXT_MODE", "full")
