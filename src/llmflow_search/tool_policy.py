"""Capability policy for live MCP tools.

JSON Schema proves that arguments have the right shape; it does not authorize the side
effect. This module is the single gate between a model-selected call and the transport.
"""

import inspect
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum


class ToolEffect(str, Enum):
    READ_ONLY = "read_only"
    LOCAL_WRITE = "local_write"
    EXTERNAL_WRITE = "external_write"
    DESTRUCTIVE = "destructive"


ToolAuthorizer = Callable[[str, dict, ToolEffect], bool | Awaitable[bool]]

_DECLARATION = re.compile(
    r"\[\s*effect\s*:\s*(read_only|local_write|external_write|destructive)\s*\]",
    re.I,
)
_DESTRUCTIVE = ("delete", "remove", "drop", "purge", "destroy", "revoke", "reset")
_READ_ONLY = (
    "search",
    "read",
    "get",
    "list",
    "fetch",
    "check",
    "validate",
    "extract",
    "detect",
    "classify",
    "resolve",
    "archive",
    "snapshot",
    "navigate",
    "crawl",
    "screenshot",
    "spec",
)
_LOCAL_WRITE = ("cache", "save", "export")
_EXTERNAL_WRITE = (
    "send",
    "post",
    "publish",
    "promote",
    "upload",
    "create",
    "update",
    "write",
    "set",
    "execute",
    "run",
)


@dataclass(frozen=True)
class ToolDecision:
    name: str
    effect: ToolEffect
    declared: bool = False


def _tool_function(name: str, tools: list[dict]) -> dict:
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if function.get("name") == name:
            return function
    return {}


def classify_tool(name: str, tools: list[dict]) -> ToolDecision:
    function = _tool_function(name, tools)
    description = str(function.get("description") or "")
    if match := _DECLARATION.search(description):
        return ToolDecision(name, ToolEffect(match.group(1).lower()), declared=True)

    lowered = name.lower()
    actions = {part for part in re.split(r"[^a-z0-9]+", lowered) if part}
    first = lowered.split("_", 1)[0]
    if actions.intersection(_DESTRUCTIVE) or first in _DESTRUCTIVE:
        effect = ToolEffect.DESTRUCTIVE
    elif actions.intersection(_READ_ONLY) or first in _READ_ONLY:
        effect = ToolEffect.READ_ONLY
    elif actions.intersection(_LOCAL_WRITE) or first in _LOCAL_WRITE:
        effect = ToolEffect.LOCAL_WRITE
    elif actions.intersection(_EXTERNAL_WRITE) or first in _EXTERNAL_WRITE:
        effect = ToolEffect.EXTERNAL_WRITE
    else:
        # An unknown arbitrary-MCP tool is not silently assumed harmless.
        effect = ToolEffect.EXTERNAL_WRITE
    return ToolDecision(name, effect)


async def authorize_tool_call(
    name: str,
    args: dict,
    tools: list[dict],
    authorizer: ToolAuthorizer | None,
) -> ToolDecision:
    decision = classify_tool(name, tools)
    if decision.effect is ToolEffect.READ_ONLY:
        return decision
    if authorizer is None:
        raise PermissionError(
            f"MCP tool '{name}' is {decision.effect.value} and no user authorization is available"
        )
    allowed = authorizer(name, args, decision.effect)
    if inspect.isawaitable(allowed):
        allowed = await allowed
    if not allowed:
        raise PermissionError(f"User denied {decision.effect.value} MCP tool '{name}'")
    return decision
