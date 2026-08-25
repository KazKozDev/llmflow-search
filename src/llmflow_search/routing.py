"""Pure production routing decisions for the research graph."""

from .config import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    MAX_EVIDENCE_ROUNDS,
    MAX_PLAN_STEPS,
    MAX_STAGNANT_ROUNDS,
)
from .search_memory import _merge_search_memory
from .state import AgentState


def route_after_execute(state: AgentState) -> str:
    """Route after each execute step — update the evidence ledger when plan is exhausted."""
    plan = state["plan"]
    completed = state["completed_steps"]
    if len(completed) >= MAX_PLAN_STEPS:
        return "evidence_ledger"
    if not plan:
        return "evidence_ledger"
    return "execute"


def route_after_evidence_ledger(state: AgentState) -> str:
    if state.get("evidence_audit", {}).get("dedup_stalled"):
        return "evidence_reextract"
    return "evidence_challenge"


def route_after_evidence_reextract(state: AgentState) -> str:
    audit = state.get("evidence_audit", {})
    if audit and audit.get("passed"):
        return "evidence_challenge"
    if state.get("sources"):
        return "answer"
    return "assimilate"


def route_after_evidence_challenge(state: AgentState) -> str:
    audit = state.get("evidence_audit", {})
    if audit.get("dedup_stalled"):
        return "evidence_reextract"
    if audit and audit.get("passed"):
        return "answer"
    verification = state.get("verification_result", {})
    if (
        verification.get("insufficient_evidence") is True
        and state.get("final_answer") == INSUFFICIENT_EVIDENCE_MESSAGE
        and not state.get("plan")
    ):
        return "answer" if state.get("sources") else "assimilate"
    if state.get("plan") and len(state.get("completed_steps", [])) < MAX_PLAN_STEPS:
        return "execute"
    if (
        state.get("evidence_round", 0) <= MAX_EVIDENCE_ROUNDS
        and len(state.get("completed_steps", [])) < MAX_PLAN_STEPS
        and state.get("stagnant_rounds", 0) < MAX_STAGNANT_ROUNDS
    ):
        return "strategy"
    # The search budget is exhausted. Supported ledger sources can still produce a
    # useful, explicitly incomplete answer; only a run with no admissible evidence
    # should terminate with the insufficient-evidence message alone.
    return "answer" if state.get("sources") else "assimilate"


def route_after_plan(state: AgentState) -> str:
    search_memory = _merge_search_memory(state.get("search_memory"))
    if (
        search_memory.get("search_exhausted")
        and state.get("final_answer") == INSUFFICIENT_EVIDENCE_MESSAGE
    ):
        return "answer" if state.get("sources") else "assimilate"
    if state["plan"]:
        return "execute"
    return "answer"


def route_after_verify(state: AgentState) -> str:
    final_answer = state.get("final_answer", "")
    draft = (
        state.get("draft_result", {})
        if isinstance(state.get("draft_result"), dict)
        else {}
    )

    has_real_answer = (
        bool(final_answer) and final_answer != INSUFFICIENT_EVIDENCE_MESSAGE
    )
    draft_answer = draft.get("answer", "") if isinstance(draft, dict) else ""
    has_valid_draft = (
        bool(draft_answer)
        and draft_answer != INSUFFICIENT_EVIDENCE_MESSAGE
        and not draft.get("insufficient_evidence")
        and bool(state.get("sources"))
    )

    if has_real_answer or has_valid_draft:
        return "assimilate"

    search_memory = _merge_search_memory(state.get("search_memory"))
    if search_memory.get("search_exhausted"):
        return "assimilate"

    if not state.get("sources"):
        if (
            state.get("evidence_round", 0) <= MAX_EVIDENCE_ROUNDS
            and len(state.get("completed_steps", [])) < MAX_PLAN_STEPS
            and state.get("stagnant_rounds", 0) < MAX_STAGNANT_ROUNDS
        ):
            return "strategy"
    return "assimilate"
