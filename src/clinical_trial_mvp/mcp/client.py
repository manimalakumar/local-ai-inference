"""Connect the analyst to the local MCP database server."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from typing import Any

from clinical_trial_mvp.config import Settings


async def call_mcp_tool(tool_name: str, arguments: dict[str, Any], settings: Settings | None = None) -> str:
    active_settings = settings or Settings.from_env()

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError as exc:
        raise RuntimeError("The mcp package is required. Install requirements.txt first.") from exc

    env = os.environ.copy()
    env["CLINICAL_TRIAL_DB_PATH"] = str(active_settings.db_path)
    env["SQL_RESULT_ROW_LIMIT"] = str(active_settings.sql_result_row_limit)
    env["SQL_EXECUTION_TIMEOUT_SECONDS"] = str(active_settings.sql_execution_timeout_seconds)
    env["PRIVACY_SALT"] = active_settings.privacy_salt
    env["PYTHONPATH"] = str(active_settings.project_root / "src") + os.pathsep + env.get("PYTHONPATH", "")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "clinical_trial_mvp.mcp.server"],
        env=env,
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)

    parts: list[str] = []
    for content in result.content:
        if getattr(content, "type", "") == "text":
            parts.append(content.text)
    return "\n".join(parts)


def call_mcp_tool_sync(tool_name: str, arguments: dict[str, Any], settings: Settings | None = None) -> str:
    """Run from a CLI or a notebook without nesting event loops.

    Create and await the coroutine in the same thread. Jupyter already owns
    the caller's loop, so its synchronous bridge needs a separate worker loop.
    """
    def run() -> str:
        return asyncio.run(call_mcp_tool(tool_name, arguments, settings))

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return run()
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="clinical-mcp") as executor:
        return executor.submit(run).result()


def execute_readonly_sql_via_mcp(sql: str, settings: Settings | None = None) -> str:
    return call_mcp_tool_sync("execute_readonly_sql", {"sql": sql}, settings)


def get_schema_via_mcp(settings: Settings | None = None) -> str:
    return call_mcp_tool_sync("get_schema", {}, settings)


def request_clarification_via_mcp(question: str, settings: Settings | None = None) -> str:
    return call_mcp_tool_sync("request_clarification", {"question": question}, settings)
