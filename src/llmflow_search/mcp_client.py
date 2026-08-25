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
    MCP_TIMEOUT_SECONDS,
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


def _compact_json_value(
    value, *, depth: int = 0, max_string_chars: int = 12000, max_items: int = 100
):
    """Bound untrusted structured output without corrupting its JSON container."""
    if depth >= 8:
        return "[nested value omitted]"
    if isinstance(value, str):
        return value[:max_string_chars]
    if isinstance(value, list):
        return [
            _compact_json_value(
                item,
                depth=depth + 1,
                max_string_chars=max_string_chars,
                max_items=max_items,
            )
            for item in value[:max_items]
        ]
    if isinstance(value, dict):
        return {
            str(key)[:200]: _compact_json_value(
                item,
                depth=depth + 1,
                max_string_chars=max_string_chars,
                max_items=max_items,
            )
            for key, item in list(value.items())[:max_items]
        }
    return value


def _bounded_content_text(text: str) -> str:
    """Clip prose directly, but parse/compact JSON before applying the wire limit."""
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return str(text)[:TOOL_RESULT_MAX_CHARS]
    # Tighten field and array budgets together until the original top-level container
    # fits. This preserves whether the server returned an object or an array.
    for max_string_chars, max_items in (
        (12000, 100),
        (6000, 50),
        (3000, 25),
        (1500, 12),
        (750, 6),
        (300, 3),
        (100, 1),
    ):
        compact = _compact_json_value(
            value, max_string_chars=max_string_chars, max_items=max_items
        )
        encoded = json.dumps(compact, ensure_ascii=False, default=str)
        if len(encoded) <= TOOL_RESULT_MAX_CHARS:
            return encoded
    # Pathological key-only objects can exceed the budget even with one field retained.
    return json.dumps(
        {"truncated": True, "preview": str(value)[:4000]}, ensure_ascii=False
    )


def _mcp_result_text(result) -> str:
    """Preserve text, embedded text resources, and structured MCP output."""
    parts: list[str] = []
    for content in getattr(result, "content", []) or []:
        text = getattr(content, "text", None)
        if isinstance(text, str) and text:
            parts.append(_bounded_content_text(text))
            continue
        resource = getattr(content, "resource", None)
        resource_text = getattr(resource, "text", None)
        if isinstance(resource_text, str) and resource_text:
            parts.append(_bounded_content_text(resource_text))

    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        structured_text = _bounded_content_text(
            json.dumps(structured, ensure_ascii=False, default=str)
        )
        if structured_text and structured_text not in parts:
            parts.append(structured_text)

    if len(parts) == 1:
        text = parts[0].strip()
    elif parts:
        decoded_parts = []
        for part in parts:
            try:
                decoded_parts.append(json.loads(part))
            except json.JSONDecodeError:
                decoded_parts = []
                break
        if decoded_parts:
            # Multiple structured content blocks are one structured MCP result. Keep the
            # aggregate valid JSON too, instead of joining objects and slicing bytes.
            text = _bounded_content_text(
                json.dumps(decoded_parts, ensure_ascii=False, default=str)
            )
        else:
            kept: list[str] = []
            used = 0
            for part in parts:
                separator = 1 if kept else 0
                if used + separator + len(part) > TOOL_RESULT_MAX_CHARS:
                    break
                kept.append(part)
                used += separator + len(part)
            text = "\n".join(kept).strip()
    else:
        text = ""
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        raise RuntimeError(text or "MCP tool reported an error")
    return text


async def load_mcp_tools() -> list[dict]:
    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=MCP_TIMEOUT_SECONDS)
            result = await asyncio.wait_for(
                session.list_tools(), timeout=MCP_TIMEOUT_SECONDS
            )
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
        result = await asyncio.wait_for(
            session.call_tool(name, args), timeout=MCP_TIMEOUT_SECONDS
        )
        return _mcp_result_text(result)

    params = StdioServerParameters(command=SERVER_CMD[0], args=SERVER_CMD[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=MCP_TIMEOUT_SECONDS)
            result = await asyncio.wait_for(
                session.call_tool(name, args), timeout=MCP_TIMEOUT_SECONDS
            )
            return _mcp_result_text(result)
