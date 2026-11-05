"""Validate and execute bounded read-only SQL against DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from threading import Timer
from typing import Any

from clinical_trial_mvp.config import Settings
from clinical_trial_mvp.data.privacy import limit_query, sanitize_rows


@dataclass(frozen=True)
class QueryResult:
    ok: bool
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    error: str | None = None


def _json_ready(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    return value


def execute_readonly_sql(sql: str, settings: Settings | None = None) -> QueryResult:
    active_settings = settings or Settings.from_env()
    try:
        import duckdb
    except ImportError as exc:
        raise RuntimeError(
            "The Python duckdb package is required. Install requirements.txt in the project venv."
        ) from exc

    limited_sql = limit_query(sql, active_settings.sql_result_row_limit)
    if active_settings.sql_execution_timeout_seconds <= 0:
        raise ValueError("SQL_EXECUTION_TIMEOUT_SECONDS must be positive.")

    try:
        with duckdb.connect(str(active_settings.db_path), read_only=True) as connection:
            timer = Timer(active_settings.sql_execution_timeout_seconds, connection.interrupt)
            timer.daemon = True
            timer.start()
            try:
                cursor = connection.execute(limited_sql)
                columns = [column[0] for column in cursor.description or []]
                raw_rows = cursor.fetchall()
            finally:
                timer.cancel()
    except Exception as exc:  # DuckDB returns several concrete exception classes.
        return QueryResult(
            ok=False,
            sql=sql,
            columns=[],
            rows=[],
            row_count=0,
            error=str(exc),
        )

    rows = [
        {column: _json_ready(value) for column, value in zip(columns, raw_row)}
        for raw_row in raw_rows
    ]
    rows = sanitize_rows(rows, active_settings.privacy_salt)
    return QueryResult(
        ok=True,
        sql=limited_sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
    )


def database_status(settings: Settings | None = None) -> dict[str, Any]:
    active_settings = settings or Settings.from_env()
    db_path = Path(active_settings.db_path)
    return {
        "db_path": str(db_path),
        "db_exists": db_path.exists(),
        "model_path": str(active_settings.model_path),
        "model_exists": active_settings.model_path.exists(),
        "model_runtime": active_settings.model_runtime,
        "ollama_deploy_model": active_settings.ollama_deploy_model,
        "ollama_model_name": active_settings.ollama_model_name,
        "csv_dir": str(active_settings.csv_dir),
        "csv_dir_exists": active_settings.csv_dir.exists(),
    }
