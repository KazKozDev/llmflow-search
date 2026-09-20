"""Answer drafting and fail-closed verification workflow."""

import json

from . import attribution, llm, report_structure, trace
from .config import (
    CITATION_COVERAGE_TARGET,
    CITATION_REPAIR,
    CITATION_REPAIR_MIN_KEPT,
    INSUFFICIENT_EVIDENCE_MESSAGE,
    REPORT_STRUCTURE_REPAIR,
    STRUCTURE_REPAIR_MIN_KEPT,
)
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

def _unconfirmed_clues(state: AgentState) -> list[str]:
    """Identifying clues the ledger never confirmed.

    They no longer block an answer (see requirements._identifying_indices), which makes
    saying so non-negotiable: an identification resting on clues nobody verified is a
    guess, and the reader is entitled to know which part of it is one.
    """
    clues: list[str] = []
    for source in ("evidence_ledger_result", "evidence_challenge_result"):
        for clue in (state.get(source) or {}).get("unconfirmed_identifying", []):
            text = str(clue).strip()
            if text and text not in clues:
                clues.append(text)
    return clues


def _dropped_gaps(state: AgentState) -> list[str]:
    """Objections the gate threw away because they named no requirement.

    An objection arrives bound to a numbered requirement, and one that cannot be bound is
    discarded so that a requirement the model invented cannot block an answer forever
    (see evidence._parse_indexed_item). The discard is deliberate and it is also blind: a
    real objection whose index the model simply failed to emit is thrown away by the same
    line, and until now it was printed to the console and recorded nowhere. "task_complete
    with three discarded objections" and "task_complete with none" were indistinguishable
    downstream, so the frequency of the second case could not be measured, and a fix for
    it could not be told from a quiet run.
    """
    gaps: list[str] = []
    for source in ("evidence_ledger_result", "evidence_challenge_result"):
        for gap in (state.get(source) or {}).get("dropped_gaps", []):
            text = str(gap).strip()
            if text and text not in gaps:
                gaps.append(text)
    return gaps


def _citation_coverage(answer: str) -> tuple[float, list[str]]:
    """Share of the answer's claim sentences that carry a citation, and the ones that do not."""
    claims = attribution.split_claims(answer)
    if not claims:
        return 1.0, []
    uncited = [claim.text for claim in claims if not claim.is_cited]
    return (len(claims) - len(uncited)) / len(claims), uncited


def _repair_citations(
    answer: str, sources_text: str, model: str, profile: Profile
) -> str:
    """Attach a source to every claim sentence that has none, or drop the sentence.

    The answer prompt has always required a citation on every factual sentence, and has
    never reliably got one: on a ten-question measured run, half of the claim sentences in
    a verified answer carried no marker. That half cannot be attributed by anyone — not by
    a reader, not by the scorer — so it caps how much of an answer can be supported no
    matter how good the evidence is.

    Asking harder in the prompt was already the behaviour. This instead re-reads the
    produced text with the same segmentation the scorer uses, names the sentences that
    failed, and spends one call on them. The result is accepted only if it actually
    improves coverage without deleting the answer — a repair that returns a stub has not
    attached citations, and the unrepaired text is better than a gutted one.
    """
    coverage, uncited = _citation_coverage(answer)
    if not CITATION_REPAIR or coverage >= CITATION_COVERAGE_TARGET or not uncited:
        return answer

    prompt = f"""SOURCES:
{sources_text}

ANSWER:
{answer}

SENTENCES WITH NO CITATION:
{json.dumps(uncited, ensure_ascii=False, indent=2)}

Return the ANSWER with each listed sentence either cited or removed."""
    with trace.model_call("citation_repair", prompt, model):
        repaired = (
            llm._ollama_chat(
                model,
                [{"role": "user", "content": prompt}],
                tools=None,
                system=profile.citation_repair,
                temperature=0,
            )
            .get("content", "")
            .strip()
        )

    repaired_coverage, _ = _citation_coverage(repaired)
    long_enough = len(repaired) >= CITATION_REPAIR_MIN_KEPT * len(answer)
    accepted = (
        bool(repaired)
        and not attribution.is_refusal(repaired)
        and long_enough
        and repaired_coverage > coverage
    )
    trace.emit(
        "citation_repair",
        coverage_before=round(coverage, 3),
        coverage_after=round(repaired_coverage, 3),
        uncited_before=len(uncited),
        chars_before=len(answer),
        chars_after=len(repaired),
        accepted=accepted,
    )
    print(
        f"  [CITE] {coverage:.0%} → {repaired_coverage:.0%} of claim sentences cited"
        + ("" if accepted else " (repair rejected, keeping the verified answer)")
    )
    return repaired if accepted else answer


def _repair_structure(
    answer: str,
    sources_text: str,
    model: str,
    profile: Profile,
    sources: list[dict] | None = None,
) -> tuple[str, list[str]]:
    """Fix the structure defects a regular expression can prove, or leave the text alone.

    The editorial contract is carried by two prompts, and a prompt is a tendency. This is
    the enforcing half: ``report_structure`` re-reads the produced report and names the
    passages that fail — an empty heading, a comparative with no figure, a point made
    twice — and one call is spent quoting them back.

    Built like ``_repair_citations`` and guarded like it, for the same reason. A repair
    call given a whole report can return a shorter, blander report that happens to trip no
    detector, which is not a repair. The result is therefore accepted only when it removes
    defects, keeps most of the text, and does not drop citation coverage on the way — a
    rewrite that cuts a cited sentence to lose a duplicate has made the report less
    attributable, and the unrepaired text is better than that.

    Returns the text to keep and the defect kinds that were still present in it.
    """
    defects = report_structure.find_defects(answer, sources)
    if not REPORT_STRUCTURE_REPAIR or not defects:
        return answer, [defect.kind for defect in defects]

    coverage_before, _ = _citation_coverage(answer)
    prompt = f"""SOURCES:
{sources_text}

REPORT:
{answer}

DEFECTS:
{chr(10).join(defect.describe() for defect in defects)}

Return the REPORT with every listed defect resolved and nothing else changed."""
    with trace.model_call("structure_repair", prompt, model):
        repaired = (
            llm._ollama_chat(
                model,
                [{"role": "user", "content": prompt}],
                tools=None,
                system=profile.structure_repair,
                temperature=0,
            )
            .get("content", "")
            .strip()
        )

    repaired_defects = (
        report_structure.find_defects(repaired, sources) if repaired else defects
    )
    coverage_after, _ = _citation_coverage(repaired) if repaired else (0.0, [])
    long_enough = len(repaired) >= STRUCTURE_REPAIR_MIN_KEPT * len(answer)
    accepted = (
        bool(repaired)
        and not attribution.is_refusal(repaired)
        and long_enough
        and len(repaired_defects) < len(defects)
        and coverage_after >= coverage_before
    )
    trace.emit(
        "structure_repair",
        defects_before=len(defects),
        defects_after=len(repaired_defects),
        kinds_before=sorted({defect.kind for defect in defects}),
        coverage_before=round(coverage_before, 3),
        coverage_after=round(coverage_after, 3),
        chars_before=len(answer),
        chars_after=len(repaired),
        accepted=accepted,
    )
    print(
        f"  [STRUCTURE] {len(defects)} → {len(repaired_defects)} defects"
        + ("" if accepted else " (repair rejected, keeping the verified answer)")
    )
    if accepted:
        return repaired, [defect.kind for defect in repaired_defects]
    return answer, [defect.kind for defect in defects]


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
    unconfirmed_clues = _unconfirmed_clues(state)
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

UNCONFIRMED_IDENTIFYING_CLUES:
{json.dumps(unconfirmed_clues, ensure_ascii=False, indent=2)}

SOURCES:
{sources_text}

Answer only from SOURCES. Use NON_BLOCKING_QUALITY_PREFERENCES only when supported.
UNCONFIRMED_IDENTIFYING_CLUES are details the asker supplied to single out the subject
which no source confirmed. They do not block the answer, but the answer must not present
the identification as settled: state the answer, then say in one short sentence which of
these clues could not be verified against the sources.
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

        # Repairs before the verdict, because the verdict is what becomes task_complete
        # and it has to describe the text that ships. Run the other way round, a repair
        # that deletes a required sentence leaves a completion verdict behind it that was
        # true of an answer nobody will read.
        #
        # Structure first, citations second: a structure repair deletes and rewrites
        # sentences, and a sentence it introduces needs the citation pass after it, not
        # before.
        structure_defects: list[str] = []
        if not insufficient:
            verified, structure_defects = _repair_structure(
                verified, sources_text, model, profile, state.get("sources")
            )
            verified = _repair_citations(verified, sources_text, model, profile)
            insufficient = attribution.is_refusal(verified)
        citation_coverage, _uncited = _citation_coverage(verified)

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
            "citation_coverage": round(citation_coverage, 3),
            "structure_defects": structure_defects,
            "dropped_gaps": _dropped_gaps(state),
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
