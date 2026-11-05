"""Apply deterministic privacy rules to clinical questions and results."""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Any

import duckdb


FORBIDDEN_SQL_WORDS = {
    "alter",
    "attach",
    "call",
    "copy",
    "create",
    "delete",
    "detach",
    "drop",
    "execute",
    "export",
    "grant",
    "import",
    "insert",
    "install",
    "load",
    "merge",
    "prepare",
    "pragma",
    "parquet_scan",
    "postgres_scan",
    "mysql_scan",
    "sqlite_scan",
    "replace",
    "revoke",
    "read_csv",
    "read_csv_auto",
    "read_json",
    "read_ndjson",
    "read_parquet",
    "read_text",
    "read_blob",
    "glob",
    "query_table",
    "set",
    "truncate",
    "update",
    "vacuum",
}

SUBJECT_ID_PATTERN = re.compile(r"\bSUBJ-\d{4,}\b", re.IGNORECASE)
COMMENT_PATTERN = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)


class QueryPolicyError(ValueError):
    """Raised when a SQL statement violates local read-only policy."""


def strip_sql_comments(sql: str) -> str:
    return COMMENT_PATTERN.sub(" ", sql)


def normalize_sql(sql: str) -> str:
    stripped = strip_sql_comments(sql).strip()
    if stripped.endswith(";"):
        stripped = stripped[:-1].strip()
    return re.sub(r"\s+", " ", stripped)


def validate_readonly_sql(sql: str) -> str:
    normalized = sql.strip().removesuffix(";").strip()
    if not normalized:
        raise QueryPolicyError("SQL query is empty.")
    try:
        statements = duckdb.extract_statements(sql)
    except duckdb.ParserException as exc:
        raise QueryPolicyError(f"SQL parser rejected query: {exc}") from exc
    if len(statements) != 1:
        raise QueryPolicyError("Only one SQL statement is allowed.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise QueryPolicyError("Only SELECT or WITH read-only queries are allowed.")

    code = _mask_sql_literals_and_comments(normalized)
    first_word_match = re.match(r"^\s*([a-zA-Z_]+)", code)
    first_word = first_word_match.group(1).lower() if first_word_match else ""
    if first_word not in {"select", "with"}:
        raise QueryPolicyError("Only SELECT or WITH read-only queries are allowed.")
    words = set(re.findall(r"\b[a-zA-Z_]+\b", code.lower()))
    forbidden = sorted(words.intersection(FORBIDDEN_SQL_WORDS))
    if forbidden:
        raise QueryPolicyError(
            "Query contains forbidden read/write keyword(s): " + ", ".join(forbidden)
        )

    if re.search(r"\bselect\s+\*", code, re.IGNORECASE) or re.search(
        r"\b[a-zA-Z_][a-zA-Z0-9_]*\.\*", code
    ):
        raise QueryPolicyError("SELECT * is not allowed. Project only the needed columns.")

    # DuckDB also accepts file readers in SELECT statements. They are outside
    # the local clinical schema even though the statement type is SELECT.
    if re.search(r"\b(from|join)\s*['\"]", normalized, re.IGNORECASE):
        raise QueryPolicyError("External or dynamic table sources are not allowed.")

    return normalized


def _mask_sql_literals_and_comments(sql: str) -> str:
    """Leave SQL structure in place while hiding string/comment contents."""
    output: list[str] = []
    index = 0
    state = "code"
    while index < len(sql):
        char = sql[index]
        next_char = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "code":
            if char == "'":
                state = "string"
                output.append(" ")
            elif char == "-" and next_char == "-":
                state = "line_comment"
                output.extend("  ")
                index += 1
            elif char == "/" and next_char == "*":
                state = "block_comment"
                output.extend("  ")
                index += 1
            else:
                output.append(char)
        elif state == "string":
            if char == "'" and next_char == "'":
                output.extend("  ")
                index += 1
            elif char == "'":
                state = "code"
                output.append(" ")
            else:
                output.append(" ")
        elif state == "line_comment":
            if char == "\n":
                state = "code"
                output.append("\n")
            else:
                output.append(" ")
        else:
            if char == "*" and next_char == "/":
                state = "code"
                output.extend("  ")
                index += 1
            else:
                output.append(" ")
        index += 1
    return "".join(output)


def limit_query(sql: str, max_rows: int) -> str:
    readonly_sql = validate_readonly_sql(sql)
    return f"SELECT * FROM ({readonly_sql}) AS readonly_result LIMIT {int(max_rows)}"


def pseudonymize_subject_id(subject_id: str, salt: str) -> str:
    digest = hmac.new(
        salt.encode("utf-8"),
        subject_id.upper().encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:12]
    return f"SUBJ_HASH_{digest}"


def sanitize_value(value: Any, salt: str) -> Any:
    if isinstance(value, str):
        if SUBJECT_ID_PATTERN.fullmatch(value):
            return pseudonymize_subject_id(value, salt)
        return SUBJECT_ID_PATTERN.sub(lambda match: pseudonymize_subject_id(match.group(0), salt), value)
    return value


def sanitize_rows(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for row in rows:
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if key.lower() == "subject_id" and value is not None:
                safe_row[key] = pseudonymize_subject_id(str(value), salt)
            else:
                safe_row[key] = sanitize_value(value, salt)
        sanitized.append(safe_row)
    return sanitized
