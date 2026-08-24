"""Public API facade.

Implementation lives in the sibling submodules. This module re-exports only the
surface that external callers (the test suite and scripts/live_smoke.py) reach for
through ``from llmflow_search import agent``. Everything else is imported from its own
submodule directly.
"""

from .mcp_client import load_mcp_tools
from .nodes import build_graph, evaluate_node, route_after_evaluate
from .observations import (
    _enrich_sources_with_observation,
    _fallback_observation,
    _normalize_observation,
    _record_observation,
)
from .prompts import (
    ANSWER_PROSE_SYSTEM_PROMPT,
    EVIDENCE_CHALLENGE_SYSTEM_PROMPT,
    EVIDENCE_LEDGER_SYSTEM_PROMPT,
    PLAN_PROMPT,
    REQUIREMENTS_SYSTEM_PROMPT,
    VERIFY_PROSE_SYSTEM_PROMPT,
    VERIFY_VERDICT_SYSTEM_PROMPT,
)
from .reports import _write_debug_report
from .search_memory import _default_search_memory, _strategy_plan_from_memory
from .sources import _audit_evidence_state, _sources_from_tool_result
from .tool_steps import _tool_call_from_step

__all__ = [
    "load_mcp_tools",
    "build_graph",
    "evaluate_node",
    "route_after_evaluate",
    "_fallback_observation",
    "_enrich_sources_with_observation",
    "_normalize_observation",
    "_record_observation",
    "ANSWER_PROSE_SYSTEM_PROMPT",
    "EVIDENCE_CHALLENGE_SYSTEM_PROMPT",
    "EVIDENCE_LEDGER_SYSTEM_PROMPT",
    "PLAN_PROMPT",
    "REQUIREMENTS_SYSTEM_PROMPT",
    "VERIFY_PROSE_SYSTEM_PROMPT",
    "VERIFY_VERDICT_SYSTEM_PROMPT",
    "_write_debug_report",
    "_default_search_memory",
    "_strategy_plan_from_memory",
    "_audit_evidence_state",
    "_sources_from_tool_result",
    "_tool_call_from_step",
]
