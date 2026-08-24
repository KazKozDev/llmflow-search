"""MCP stdio client: tool-schema loading and tool invocation."""

import asyncio
import json
import random
import re
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from . import trace
from .config import (
    SEARCH_DELAY_JITTER,
    SEARCH_GROUP_DELAY_SECONDS,
    SERVER_CMD,
    TOOL_RESULT_MAX_CHARS,
)
from .console import print

# Rate-limited backends, grouped by who actually throttles the request. Tools in one
# group share a slot; separate groups never wait on each other, so a scraper pause does
# not stall an API call.
_SEARCH_TOOL_GROUPS = {
    "web_search": "scraper",
    "web_deep_search": "scraper",
    "web_search_recent": "scraper",
    "github_search": "api",
    "papers_search": "api",
    "encyclopedia_search": "api",
    "archive_search": "archive",
    "web_archive_fetch": "archive",
}
_search_request_locks: dict[str, asyncio.Lock] = {}
_last_search_request_started_at: dict[str, float] = {}
_monotonic = time.monotonic
_random = random.random

# How a server declares its own pacing, in the tool description list_tools returns:
# "[throttle: api]", or "[throttle: none]" for a tool nobody meters. Which backend a
# tool actually talks to is the server's knowledge, not ours — the name map above is a
# default for footnote-mcp, and it was wrong for every other server exposing a tool
# called web_search, including a local one reading JSON off disk.
_THROTTLE_DECLARATION = re.compile(r"\[\s*throttle\s*:\s*([a-z0-9_]+)\s*\]", re.I)
_declared_throttle_groups: dict[str, str | None] = {}


def _declared_throttle_group(function: dict) -> tuple[bool, str | None]:
    """The pacing a tool declares for itself: (declared?, group or None for unmetered)."""
    match = _THROTTLE_DECLARATION.search(str(function.get("description") or ""))
    if not match:
        return False, None
    group = match.group(1).lower()
    return True, None if group in ("none", "off", "unmetered") else group


def configure_tool_pacing(tools: list[dict]) -> None:
    """Adopt the pacing the connected server declares, replacing any earlier server's.

    Called from ``_tool_schema_list`` so every consumer of a live catalog — the app, the
    benchmark harness, the smoke script — is paced by the server it actually connected to.
    """
    _declared_throttle_groups.clear()
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name") or "").strip()
        declared, group = _declared_throttle_group(function)
        if name and declared:
            _declared_throttle_groups[name] = group


def _throttle_group(name: str) -> str | None:
    """The backend family a tool draws on, or None when it is not rate-limited."""
    if name in _declared_throttle_groups:
        return _declared_throttle_groups[name]
    return _SEARCH_TOOL_GROUPS.get(name)


async def _wait_for_search_request_slot(name: str) -> None:
    """Keep rate-limited MCP calls separated across batches and retries."""
    group = _throttle_group(name)
    if group is None:
        return
    delay = SEARCH_GROUP_DELAY_SECONDS.get(group, 0.0)
    if delay <= 0:
        return
    delay += delay * SEARCH_DELAY_JITTER * _random()

    lock = _search_request_locks.setdefault(group, asyncio.Lock())
    async with lock:
        now = _monotonic()
        remaining = delay - (now - _last_search_request_started_at.get(group, 0.0))
        if remaining > 0:
            print(
                f"    [wait] {remaining:.0f}s before next {group} request", flush=True
            )
            await asyncio.sleep(remaining)
            now = _monotonic()
        _last_search_request_started_at[group] = now


def _tool_schema_list(result) -> list[dict]:
    tools = [
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
    configure_tool_pacing(tools)
    return tools


def _format_tool_catalog(tools: list[dict]) -> str:
    """Render every live MCP tool compactly for schema-constrained planning."""
    lines = []
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_schema = function.get("parameters")
        schema: dict = raw_schema if isinstance(raw_schema, dict) else {}
        raw_properties = schema.get("properties")
        properties: dict = raw_properties if isinstance(raw_properties, dict) else {}
        raw_required = schema.get("required")
        required = set(raw_required if isinstance(raw_required, list) else [])
        params = []
        for param_name, raw_spec in properties.items():
            spec = raw_spec if isinstance(raw_spec, dict) else {}
            param_type = spec.get("type", "any")
            if isinstance(param_type, list):
                param_type = "|".join(str(item) for item in param_type)
            marker = "*" if param_name in required else ""
            params.append(f"{param_name}{marker}:{param_type}")
        description = " ".join(str(function.get("description") or "").split())[:240]
        lines.append(f"- {name}({', '.join(params)}): {description}")
    return "\n".join(lines) or "(no MCP tools available)"


def _mcp_result_text(result) -> str:
    """Preserve text, embedded text resources, and structured MCP output."""
    parts: list[str] = []
    for content in getattr(result, "content", []) or []:
        text = getattr(content, "text", None)
        if isinstance(text, str) and text:
            parts.append(text)
            continue
        resource = getattr(content, "resource", None)
        resource_text = getattr(resource, "text", None)
        if isinstance(resource_text, str) and resource_text:
            parts.append(resource_text)

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        structured_text = json.dumps(structured, ensure_ascii=False, default=str)
        if structured_text and structured_text not in parts:
            parts.append(structured_text)

    text = "\n".join(parts).strip()
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        raise RuntimeError(text or "MCP tool reported an error")
    return text[:TOOL_RESULT_MAX_CHARS]


async def load_mcp_tools() -> list[dict]:
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return _tool_schema_list(result)


async def _call_mcp_tool(
    name: str, args: dict, session: ClientSession | None = None
) -> str:
    """Execute one MCP tool, return result text."""
    # Waiting for a rate-limit slot is separated from the call it precedes: on a metered
    # provider the wait dominates, and a time breakdown that folds the two together
    # cannot say whether a slow run was slow or merely paced.
    with trace.span("throttle_wait", tool=name, group=_throttle_group(name)):
        await _wait_for_search_request_slot(name)

    if session is not None:
        result = await session.call_tool(name, args)
        return _mcp_result_text(result)

    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, args)
            return _mcp_result_text(result)
