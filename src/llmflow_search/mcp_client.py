"""MCP stdio client: tool-schema loading and tool invocation."""

import asyncio
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import SEARCH_REQUEST_DELAY_SECONDS, SERVER_CMD, TOOL_RESULT_MAX_CHARS

_SEARCH_TOOLS = {"web_search", "web_deep_search"}
_search_request_lock = asyncio.Lock()
_last_search_request_started_at = 0.0
_monotonic = time.monotonic


async def _wait_for_search_request_slot(name: str) -> None:
    """Keep search-engine-backed MCP calls separated across batches and retries."""
    global _last_search_request_started_at

    if name not in _SEARCH_TOOLS or SEARCH_REQUEST_DELAY_SECONDS <= 0:
        return

    async with _search_request_lock:
        now = _monotonic()
        remaining = SEARCH_REQUEST_DELAY_SECONDS - (now - _last_search_request_started_at)
        if remaining > 0:
            await asyncio.sleep(remaining)
            now = _monotonic()
        _last_search_request_started_at = now


def _tool_schema_list(result) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in result.tools
    ]


async def load_mcp_tools() -> list[dict]:
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return _tool_schema_list(result)


async def _call_mcp_tool(name: str, args: dict, session: ClientSession | None = None) -> str:
    """Execute one MCP tool, return result text."""
    await _wait_for_search_request_slot(name)

    if session is not None:
        result = await session.call_tool(name, args)
        text = ""
        for c in result.content:
            if hasattr(c, "text") and isinstance(c.text, str):  # type: ignore[union-attr]
                text += c.text  # type: ignore[union-attr]
        return text[:TOOL_RESULT_MAX_CHARS]

    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            text = ""
            for c in result.content:
                if hasattr(c, "text") and isinstance(c.text, str):  # type: ignore[union-attr]
                    text += c.text  # type: ignore[union-attr]
            return text[:TOOL_RESULT_MAX_CHARS]
