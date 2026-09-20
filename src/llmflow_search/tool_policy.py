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
    # Verbs that read evidence and return a judgement about it. Their absence here was
    # not a decision: an unrecognised verb falls through to external_write, so the
    # server's own fact-checking tools — corroborate_claim, evidence_entailment,
    # locate_claim_span, reconcile_time_series — were gated behind a permission prompt
    # while changing nothing anywhere. Measured against the live server, none of its 45
    # tools declares an effect, so every one of these verdicts comes from this list.
    "corroborate",
    "entail",
    "entailment",
    "locate",
    "reconcile",
    "parse",
    "scroll",
    "diagnose",
    "health",
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
    elif first == "browser":
        # A browser tool drives a local session: setting its date range or its viewport
        # changes what the next read returns and nothing beyond this machine. Reading
        # "set" as an external write left browser_set_date_range behind a prompt while
        # browser_extract_tables_for_date_range, which is useless without it, ran freely.
        effect = ToolEffect.LOCAL_WRITE
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


# Effects an unattended run may perform without anyone to ask. Reading is always allowed
# (it never reaches an authorizer); local writes touch this machine only — a cache entry,
# a browser's date range. Anything that leaves the machine, and anything that executes
# code, still needs a person.
UNATTENDED_EFFECTS = frozenset({ToolEffect.READ_ONLY, ToolEffect.LOCAL_WRITE})


def unattended_authorizer(
    allowed: frozenset[ToolEffect] = UNATTENDED_EFFECTS,
) -> ToolAuthorizer:
    """An authorizer for runs with no user: a stated policy instead of a missing one.

    Passing no authorizer at all is itself a policy — refuse everything above read_only —
    and it was the one the benchmark ran under, silently. That measured an agent nobody
    uses: the interactive CLI asks and the user says yes, so the benchmark's agent had
    strictly fewer capabilities than the shipped one, including the file-parsing path that
    turns a PDF or a spreadsheet into evidence.

    Stating the policy makes the difference visible and adjustable, rather than a
    consequence of an argument nobody passed.
    """

    def authorize(name: str, args: dict, effect: ToolEffect) -> bool:
        return effect in allowed

    return authorize
