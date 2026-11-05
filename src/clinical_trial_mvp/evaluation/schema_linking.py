"""Score reference table and qualified-column coverage in supplied schema context."""

from __future__ import annotations

import re

from clinical_trial_mvp.data.schema import TABLE_SCHEMAS


def reference_schema_links(sql: str) -> tuple[set[str], dict[str, set[str]]]:
    aliases: dict[str, str] = {}
    tables: set[str] = set()
    for match in re.finditer(r"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?", sql, re.I):
        table = match.group(1).lower()
        if table not in TABLE_SCHEMAS:
            continue
        tables.add(table)
        aliases[table] = table
        alias = (match.group(2) or "").lower()
        if alias and alias not in {"on", "where", "left", "right", "inner", "outer", "join", "group", "order", "limit"}:
            aliases[alias] = table
    columns: dict[str, set[str]] = {table: set() for table in tables}
    for alias, column in re.findall(r"\b(\w+)\.(\w+)\b", sql):
        table = aliases.get(alias.lower())
        if table and column.lower() in {item.name for item in TABLE_SCHEMAS[table]}:
            columns[table].add(column.lower())
    without_literals = re.sub(r"'(?:''|[^'])*'", "", sql)
    for table in tables:
        for item in TABLE_SCHEMAS[table]:
            if re.search(rf"\b{re.escape(item.name)}\b", without_literals, re.I):
                owners = [name for name in tables if item.name in {column.name for column in TABLE_SCHEMAS[name]}]
                if len(owners) == 1:
                    columns[table].add(item.name)
    return tables, columns


def schema_linking_score(reference_sql: str | None, selection: dict | None) -> dict | None:
    if not reference_sql or not selection:
        return None
    needed_tables, needed_columns = reference_schema_links(reference_sql)
    selected_tables = set(selection.get("tables", []))
    selected_columns = {table: set(columns) for table, columns in selection.get("columns", {}).items()}
    missing_tables = sorted(needed_tables - selected_tables)
    missing_columns = sorted(
        f"{table}.{column}" for table, columns in needed_columns.items()
        for column in columns - selected_columns.get(table, set())
    )
    return {
        "covered": not missing_tables and not missing_columns,
        "missing_tables": missing_tables,
        "missing_qualified_columns": missing_columns,
        "scope": "Reference tables and qualified columns; unqualified columns and business meaning require semantic review.",
    }
