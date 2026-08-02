"""The LangGraph AgentState shape."""

from typing import TypedDict


class AgentState(TypedDict):
    task: str
    conversation_context: str  # prior Q&A history — injected into plan only
    requirements_result: dict
    plan: list[str]          # remaining plan steps (not yet executed)
    completed_steps: list[dict]  # [{step, result, tools_used}]
    scratchpad: str          # raw collected data
    candidate_sources: list[dict]  # fetched sources before admissibility review
    admissible_sources: list[dict] # sources tied to supported evidence-ledger claims
    sources: list[dict]      # admitted sources used as answer evidence
    evidence_ledger_result: dict
    evidence_challenge_result: dict
    draft_result: dict       # structured answer before verification
    verification_result: dict
    evidence_audit: dict
    final_answer: str
    iteration: int
    replan_count: int        # ponytail: hard cap at 3 replans. upgrade: configurable limit when agent gets stuck on legitimate multi-step tasks.
    evidence_round: int      # additional search rounds after insufficient evidence
    search_memory: dict
    answer_mode: str         # "strict" | "roundup" — decided once by requirements_node, stable across replans
    stagnant_rounds: int      # consecutive evidence rounds with no increase in supported_claim_count
    last_supported_claim_count: int  # previous round's supported_claim_count, for stagnation comparison
