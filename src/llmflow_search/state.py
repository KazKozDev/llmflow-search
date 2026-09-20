"""The LangGraph AgentState shape."""

from typing import TypedDict


class AgentState(TypedDict):
    task: str
    conversation_context: str  # prior Q&A history — injected into plan only
    requirements_result: dict
    plan: list[str]  # remaining plan steps (not yet executed)
    completed_steps: list[dict]  # [{step, result, tools_used}]
    scratchpad: str  # raw collected data
    candidate_sources: list[dict]  # fetched sources before admissibility review
    admissible_sources: list[dict]  # sources tied to supported evidence-ledger claims
    sources: list[dict]  # admitted sources used as answer evidence
    evidence_ledger_result: dict
    evidence_challenge_result: dict
    draft_result: dict  # structured answer before verification
    verification_result: dict
    evidence_audit: dict
    # Objections that blocked this run once, under a usable requirement index. A later
    # round that raises the same objection without an index is not inventing it — see
    # evidence._recurring_gap.
    enforced_gaps: list[str]
    # One entry per condition the question sets, with its status, the pages that settled
    # it and the claim they support. A projection of the evidence ledger — see
    # constraints.py — so it cannot disagree with the ledger it is read from.
    constraint_registry: list[dict]
    final_answer: str
    iteration: int
    evidence_round: int  # additional search rounds after insufficient evidence
    search_memory: dict
    answer_mode: str  # "strict" | "roundup" — decided once by requirements_node, stable across replans
    stagnant_rounds: int  # consecutive evidence rounds with no increase in supported_claim_count
    last_supported_claim_count: int  # previous round's supported_claim_count, for stagnation comparison
