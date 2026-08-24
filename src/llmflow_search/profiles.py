"""Server profiles — what differs between the footnote-mcp server and any other MCP server.

A profile bundles the prompts and two hook functions that the graph nodes vary per server.
``FOOTNOTE_PROFILE`` supplies the research-specific prompts and legacy step parser.
``GENERIC_PROFILE`` supplies tool-agnostic prompts and treats arbitrary tool output as evidence.
Both profiles first resolve exact calls through the live JSON Schemas returned by ``list_tools``;
their hooks are compatibility fallbacks. ``select_profile`` picks one from the connected server's
tool list (honoring the ``LLMFLOW_SEARCH_PROFILE`` env override).
"""

import os
from collections.abc import Callable
from dataclasses import dataclass

from . import prompts
from .sources import _sources_from_tool_result, generic_sources_from_tool_result
from .tool_steps import _tool_call_from_step


@dataclass(frozen=True)
class Profile:
    name: str
    # prompts
    requirements: str
    plan: str
    execute: str
    eval: str
    answer_prose: str
    verify_prose: str
    verify_verdict: str
    evidence_ledger: str
    evidence_challenge: str
    strategy: str
    post_batch: str
    # hooks
    tool_call_from_step: Callable[[str], dict | None]
    # (tool_name, tool_result, context) -> sources. ``context`` carries run state a tool
    # result cannot supply on its own, such as the current browser page's address.
    sources_from_tool_result: Callable[[str, str, dict | None], list]
    fallback_step: Callable[[str], str]
    # whether the footnote search-strategy/replan machinery applies
    uses_search_memory: bool


FOOTNOTE_PROFILE = Profile(
    name="footnote",
    requirements=prompts.REQUIREMENTS_SYSTEM_PROMPT,
    plan=prompts.PLAN_PROMPT,
    execute=prompts.EXECUTE_PROMPT,
    eval=prompts.EVAL_PROMPT,
    answer_prose=prompts.ANSWER_PROSE_SYSTEM_PROMPT,
    verify_prose=prompts.VERIFY_PROSE_SYSTEM_PROMPT,
    verify_verdict=prompts.VERIFY_VERDICT_SYSTEM_PROMPT,
    evidence_ledger=prompts.EVIDENCE_LEDGER_SYSTEM_PROMPT,
    evidence_challenge=prompts.EVIDENCE_CHALLENGE_SYSTEM_PROMPT,
    strategy=prompts.STRATEGY_SYSTEM_PROMPT,
    post_batch=prompts.POST_BATCH_PROMPT,
    tool_call_from_step=_tool_call_from_step,
    sources_from_tool_result=_sources_from_tool_result,
    fallback_step=lambda task: f"web_search: {task}",
    uses_search_memory=True,
)

GENERIC_PROFILE = Profile(
    name="generic",
    # requirements / execute / answer / verify prompts are already tool-agnostic — reused.
    requirements=prompts.REQUIREMENTS_SYSTEM_PROMPT,
    plan=prompts.GENERIC_PLAN_PROMPT,
    execute=prompts.EXECUTE_PROMPT,
    eval=prompts.GENERIC_EVAL_PROMPT,
    answer_prose=prompts.ANSWER_PROSE_SYSTEM_PROMPT,
    verify_prose=prompts.VERIFY_PROSE_SYSTEM_PROMPT,
    verify_verdict=prompts.VERIFY_VERDICT_SYSTEM_PROMPT,
    evidence_ledger=prompts.EVIDENCE_LEDGER_SYSTEM_PROMPT,
    evidence_challenge=prompts.EVIDENCE_CHALLENGE_SYSTEM_PROMPT,
    strategy=prompts.STRATEGY_SYSTEM_PROMPT,  # unused: generic strategy replans from scratch
    post_batch=prompts.GENERIC_POST_BATCH_PROMPT,
    tool_call_from_step=lambda step: None,  # live-schema resolver handles exact generic steps
    sources_from_tool_result=generic_sources_from_tool_result,
    fallback_step=lambda task: task,
    uses_search_memory=False,
)

# A server is "footnote" when it exposes the signature web-research tools.
FOOTNOTE_SIGNATURE = {"web_search", "web_read"}


def select_profile(tool_names, env: str | None = None) -> Profile:
    """Choose a profile from the server's tool names. Honors LLMFLOW_SEARCH_PROFILE
    (`footnote` | `generic` | `auto`, default `auto`)."""
    choice = (
        (env if env is not None else os.getenv("LLMFLOW_SEARCH_PROFILE", "auto"))
        .strip()
        .lower()
    )
    if choice == "footnote":
        return FOOTNOTE_PROFILE
    if choice == "generic":
        return GENERIC_PROFILE
    return (
        FOOTNOTE_PROFILE if FOOTNOTE_SIGNATURE <= set(tool_names) else GENERIC_PROFILE
    )
