"""Answer drafting and fail-closed verification workflow."""

import json

from . import llm, trace
from .config import INSUFFICIENT_EVIDENCE_MESSAGE
from .console import print
from .llm import _json_loads_best_effort
from .profiles import Profile
from .requirements import _normalize_requirements, _proof_requirements
from .sources import _effective_question, _format_sources_for_llm
from .state import AgentState

VERIFY_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "missing": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
        "coverage_complete": {"type": "boolean"},
    },
    "required": ["missing", "notes", "coverage_complete"],
    "additionalProperties": False,
}

async def answer_node(
    state: AgentState, model: str, _tools: list[dict], profile: Profile
) -> dict:
    """Draft a source-grounded structured answer."""
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    proof_requirements = _proof_requirements(requirements)
    quality_preferences = list(requirements.get("quality_preferences", []))
    sources_text, valid_source_ids = _format_sources_for_llm(
        state.get("sources", []), question
    )
    audit = state.get("evidence_audit", {}) or {}
    partial_answer = audit.get("passed") is False
    evidence_gaps = [str(gap) for gap in audit.get("gaps", []) if str(gap).strip()]
    iteration = state["iteration"] + 1

    answer_kind = "partial answer" if partial_answer else "answer"
    print(
        f"\n  [ANSWER] Drafting {answer_kind} from {len(valid_source_ids)} sources..."
    )
    if not valid_source_ids:
        draft = {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "claims": [],
            "coverage": {
                "requirements_addressed": [],
                "overall_status": "missing",
            },
            "insufficient_evidence": True,
        }
        return {
            "draft_result": draft,
            "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "iteration": iteration,
        }

    prompt = f"""QUESTION:
{question}

PROOF_REQUIREMENTS:
{json.dumps(proof_requirements, ensure_ascii=False, indent=2)}

NON_BLOCKING_QUALITY_PREFERENCES:
{json.dumps(quality_preferences, ensure_ascii=False, indent=2)}

PARTIAL_ANSWER_MODE:
{json.dumps(partial_answer)}

KNOWN_EVIDENCE_GAPS:
{json.dumps(evidence_gaps, ensure_ascii=False, indent=2)}

SOURCES:
{sources_text}

Answer only from SOURCES. Use NON_BLOCKING_QUALITY_PREFERENCES only when supported.
If PARTIAL_ANSWER_MODE is false, satisfy PROOF_REQUIREMENTS.
If PARTIAL_ANSWER_MODE is true, give the most useful supported partial answer instead of refusing:
- state clearly at the beginning that the answer is partial;
- include every useful supported finding relevant to the question;
- briefly identify the relevant KNOWN_EVIDENCE_GAPS;
- never fill a gap with outside knowledge or an inference."""

    # Prose-first: rich, reliable prose with inline [n] citations. A large structured
    # JSON envelope is fragile on content-heavy answers, so we draft prose and verify
    # the prose against the same bounded source set (see verify_node).
    with trace.model_call("answer", prompt, model):
        prose = (
            llm._ollama_chat(
                model,
                [{"role": "user", "content": prompt}],
                tools=None,
                system=profile.answer_prose,
            )
            .get("content", "")
            .strip()
        )

    if prose and prose != INSUFFICIENT_EVIDENCE_MESSAGE:
        draft = {
            "answer": prose,
            "claims": [],
            "coverage": {"requirements_addressed": [], "overall_status": "partial"},
            "insufficient_evidence": False,
            "salvaged_prose": True,
        }
    else:
        draft = {
            "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "claims": [],
            "coverage": {"requirements_addressed": [], "overall_status": "missing"},
            "insufficient_evidence": True,
        }

    return {"draft_result": draft, "iteration": iteration}


async def verify_node(
    state: AgentState, model: str, _tools: list[dict], profile: Profile
) -> dict:
    """Verify the drafted answer against the same bounded source set."""
    question = _effective_question(state["task"])
    requirements = _normalize_requirements(state.get("requirements_result"), question)
    proof_requirements = _proof_requirements(requirements)
    quality_preferences = list(requirements.get("quality_preferences", []))
    sources_text, valid_source_ids = _format_sources_for_llm(
        state.get("sources", []), question
    )
    draft = state.get("draft_result", {})
    audit = state.get("evidence_audit", {}) or {}
    partial_answer = audit.get("passed") is False
    evidence_gaps = [str(gap) for gap in audit.get("gaps", []) if str(gap).strip()]
    iteration = state["iteration"] + 1

    print("  [VERIFY] Checking claims against sources...")
    if draft.get("salvaged_prose") is True and valid_source_ids:
        # Prose answer → verify in prose: model re-checks every statement and inline
        # citation against the bounded source set and returns a corrected prose answer.
        # Prose-in/prose-out avoids the fragile JSON envelope while keeping real grounding.
        print("  [VERIFY] Grounding prose answer against sources...")
        verify_prompt = f"""PROOF_REQUIREMENTS:
{json.dumps(proof_requirements, ensure_ascii=False, indent=2)}

NON_BLOCKING_QUALITY_PREFERENCES:
{json.dumps(quality_preferences, ensure_ascii=False, indent=2)}

PARTIAL_ANSWER_MODE:
{json.dumps(partial_answer)}

KNOWN_EVIDENCE_GAPS:
{json.dumps(evidence_gaps, ensure_ascii=False, indent=2)}

SOURCES:
{sources_text}

DRAFT_ANSWER:
{draft.get("answer", "")}

Return the corrected, fully-grounded answer. In PARTIAL_ANSWER_MODE, preserve useful
supported findings and the explicit partial-answer disclosure; do not reject the whole
answer merely because KNOWN_EVIDENCE_GAPS remain."""
        with trace.model_call("verify", verify_prompt, model):
            verified = (
                llm._ollama_chat(
                    model,
                    [{"role": "user", "content": verify_prompt}],
                    tools=None,
                    system=profile.verify_prose,
                )
                .get("content", "")
                .strip()
            )

        # A missing verifier result is not evidence that the draft was grounded. Preserve
        # fail-closed status; callers may still inspect draft_result in a debug report.
        verifier_failed = not verified
        if verifier_failed:
            verified = INSUFFICIENT_EVIDENCE_MESSAGE
        insufficient = (not verified) or verified == INSUFFICIENT_EVIDENCE_MESSAGE

        # Small JSON verdict: a compact, machine-readable quality report about the prose
        # answer. No answer text inside, so it parses reliably (unlike the old big envelope).
        coverage_complete = False
        missing: list[str] = []
        verdict_notes: list[str] = []
        if not insufficient:
            verdict_prompt = f"""PROOF_REQUIREMENTS:
{json.dumps(proof_requirements, ensure_ascii=False, indent=2)}

NON_BLOCKING_QUALITY_PREFERENCES:
{json.dumps(quality_preferences, ensure_ascii=False, indent=2)}

SOURCES:
{sources_text}

FINAL_ANSWER:
{verified}

Return the compact JSON verdict."""
            with trace.model_call("verify_verdict", verdict_prompt, model):
                verdict_raw = llm._ollama_chat_schema(
                    model,
                    [{"role": "user", "content": verdict_prompt}],
                    system=profile.verify_verdict,
                    format_schema=VERIFY_VERDICT_SCHEMA,
                )
            verdict = _json_loads_best_effort(verdict_raw, None)
            expected_verdict_keys = {"coverage_complete", "missing", "notes"}
            verdict_valid = (
                isinstance(verdict, dict)
                and set(verdict) == expected_verdict_keys
                and isinstance(verdict.get("coverage_complete"), bool)
                and isinstance(verdict.get("missing"), list)
                and isinstance(verdict.get("notes"), list)
                and all(isinstance(item, str) for item in verdict["missing"])
                and all(isinstance(item, str) for item in verdict["notes"])
            )
            if verdict_valid:
                assert isinstance(verdict, dict)
                coverage_complete = verdict["coverage_complete"]
                missing = [str(m) for m in verdict.get("missing", []) if m]
                verdict_notes = [str(n) for n in verdict.get("notes", []) if n]
            else:
                missing = ["Verification verdict was missing or malformed."]
                verdict_notes = ["The answer was not marked complete."]
            if missing:
                print(f"  [VERIFY] Coverage gaps: {', '.join(missing[:4])}")

        verification = {
            "final_answer": verified,
            "claim_checks": [],
            "task_complete": (not insufficient) and coverage_complete,
            "coverage": {
                "overall_status": "complete" if coverage_complete else "partial"
            },
            "gaps": missing,
            "notes": verdict_notes,
            "insufficient_evidence": insufficient,
        }
        update = {
            "verification_result": verification,
            "final_answer": verified,
            "iteration": iteration,
        }
        if insufficient:
            update["evidence_round"] = state.get("evidence_round", 0) + 1
        return update
    if not valid_source_ids or draft.get("insufficient_evidence") is True:
        next_round = state.get("evidence_round", 0) + 1
        verification = {
            "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "claim_checks": [],
            "task_complete": False,
            "coverage": {
                "requirements_addressed": [],
                "overall_status": "missing",
            },
            "gaps": ["No source-backed draft answer is available."],
            "insufficient_evidence": True,
        }
        return {
            "verification_result": verification,
            "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "evidence_round": next_round,
            "iteration": iteration,
        }

    # Unreachable in normal flow: answer_node always produces either a salvaged_prose
    # draft (handled above) or an insufficient_evidence draft (handled above). This is a
    # safety net for any unexpected draft shape.
    verification = {
        "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "task_complete": False,
        "gaps": ["No usable answer draft."],
        "insufficient_evidence": True,
    }
    return {
        "verification_result": verification,
        "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "evidence_round": state.get("evidence_round", 0) + 1,
        "iteration": iteration,
    }
