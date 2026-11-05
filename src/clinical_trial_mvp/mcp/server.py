"""Expose schema and read-only query tools over local MCP."""

from __future__ import annotations

import json

from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.data.db import execute_readonly_sql as run_db_query
from clinical_trial_mvp.agent.clarification import clarification_for
from clinical_trial_mvp.data.schema import SCHEMA_PROMPT


try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only before dependencies are installed.
    raise RuntimeError("The mcp package is required. Install requirements.txt first.") from exc


mcp = FastMCP("clinical-trial-duckdb")


@mcp.tool()
def get_schema() -> str:
    """Return the clinical trial DuckDB schema and table relationships."""
    return SCHEMA_PROMPT


@mcp.tool()
def execute_readonly_sql(sql: str) -> str:
    """Run one read-only DuckDB SQL query and return JSON with sanitized rows."""
    settings = Settings.from_env()
    try:
        result = run_db_query(sql, settings)
        return json.dumps(
            {
                "ok": result.ok,
                "sql": result.sql,
                "columns": result.columns,
                "rows": result.rows,
                "row_count": result.row_count,
                "error": result.error,
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps(
            {
                "ok": False,
                "sql": sql,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": str(exc),
            },
            indent=2,
        )


@mcp.tool()
def request_clarification(question: str) -> str:
    """Return a clarification question for ambiguous clinical-operations terms."""
    clarification = clarification_for(question)
    return json.dumps(
        {"clarification_required": clarification is not None, "question": clarification}
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
