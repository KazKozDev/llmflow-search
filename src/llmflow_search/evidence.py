"""Deterministic evidence-ledger schemas and normalization contracts."""

from .attribution import content_terms
from .config import ROUNDUP_MIN_CLAIMS, ROUNDUP_MIN_SOURCES

# How much of its wording an unindexed objection must share with one that already blocked
# this run before it counts as the same objection returning without its index.
#
# Shared content words rather than character similarity, because the gates re-word rather
# than repeat: "No primary source confirms the 1913 figure" comes back as "The 1913 figure
# is confirmed by no primary source", which is the same objection, scores 0.47 as a string
# and 0.71 as a bag of content words. Reordering is what a rewrite does most.
#
# The errors are not symmetric, which is what puts the threshold this low. A false match
# keeps a run searching for something it may already hold — slower, still correct. A miss
# lets a real requirement disappear and stamps the answer complete, which is the failure
# being fixed here.
GAP_RECURRENCE_OVERLAP = 0.6


def _same_gap(text: str, earlier: str) -> bool:
    terms, earlier_terms = set(content_terms(text)), set(content_terms(earlier))
    if not terms or not earlier_terms:
        return False
    return (
        len(terms & earlier_terms) / len(terms | earlier_terms)
    ) >= GAP_RECURRENCE_OVERLAP


def _recurring_gap(text: str, enforced_gaps) -> bool:
    """Whether this objection already blocked the run once, under a usable index.

    An objection with no index is normally discarded, because a requirement the model
    invented must not be able to block an answer forever. That reasoning holds only for an
    objection nobody has seen before. One that blocked an earlier round with a valid index
    is not an invention — it is a real requirement whose index the model failed to emit
    this time, and discarding it is how a run that was correctly blocked becomes complete
    without anything being found.
    """
    return any(_same_gap(text, str(earlier)) for earlier in (enforced_gaps or ()))


def _numbered_requirements_block(completion_criteria: list[str]) -> str:
    if not completion_criteria:
        return "(none)"
    return "\n".join(f"{i}. {c}" for i, c in enumerate(completion_criteria))


def _parse_indexed_item(item, requirement_count: int) -> tuple[str, int | None]:
    """Parse a {requirement_index, text} gap item against the numbered PROOF_REQUIREMENTS list.

    requirement_index is schema-required (see _evidence_ledger_schema/_evidence_challenge_schema),
    but decoding is only truly enforced on GGUF-backed models (see llm._schema_capable_model) — this
    is a defensive parse for whatever the model actually returned, not the primary correctness
    mechanism. A missing/out-of-range index means the item cannot be tied to a real requirement.
    """
    if isinstance(item, dict):
        text = str(item.get("text") or "").strip()[:300]
        raw_idx = item.get("requirement_index")
        try:
            idx = int(raw_idx) if raw_idx is not None else None
        except (TypeError, ValueError):
            idx = None
    elif isinstance(item, str):
        text, idx = item.strip()[:300], None
    else:
        text, idx = "", None
    if idx is None or not (0 <= idx < requirement_count):
        idx = None
    return text, idx


def _evidence_ledger_schema(requirement_count: int) -> dict:
    max_index = max(requirement_count - 1, 0)
    indexed_gap = {
        "type": "object",
        "properties": {
            "requirement_index": {
                "type": "integer",
                "minimum": 0,
                "maximum": max_index,
            },
            "text": {"type": "string"},
        },
        "required": ["requirement_index", "text"],
    }
    # Property order is generation order under constrained decoding: the verdict comes
    # last so it is written after the rows and reasoning that justify it, not before.
    return {
        "type": "object",
        "properties": {
            "ledger": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "requirement_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max_index,
                        },
                        "requirement": {"type": "string"},
                        "proposed_claim": {"type": "string"},
                        "event_date": {"type": "string"},
                        "publication_date": {"type": "string"},
                        "location": {"type": "string"},
                        "source_ids": {"type": "array", "items": {"type": "integer"}},
                        "source_quality": {"type": "string"},
                        "support_status": {
                            "type": "string",
                            "enum": ["supported", "partial", "missing", "rejected"],
                        },
                        "support_level": {
                            "type": "string",
                            "enum": ["supported", "partial", "missing"],
                        },
                        "can_use_in_answer": {"type": "boolean"},
                        "missing": {"type": "string"},
                        "rejection_reason": {"type": "string"},
                    },
                    "required": [
                        "claim_id",
                        "requirement_index",
                        "support_status",
                        "support_level",
                        "can_use_in_answer",
                    ],
                },
            },
            "global_missing": {"type": "array", "items": indexed_gap},
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
            "answer_ready": {"type": "boolean"},
        },
        "required": [
            "ledger",
            "global_missing",
            "next_steps",
            "reason",
            "answer_ready",
        ],
    }


def _evidence_challenge_schema(requirement_count: int) -> dict:
    max_index = max(requirement_count - 1, 0)
    return {
        "type": "object",
        "properties": {
            "blocking_gaps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "requirement_index": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": max_index,
                        },
                        "text": {"type": "string"},
                    },
                    "required": ["requirement_index", "text"],
                },
            },
            "next_steps": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
            "answer_permitted": {"type": "boolean"},
        },
        "required": ["blocking_gaps", "next_steps", "reason", "answer_permitted"],
    }


def _normalize_evidence_ledger_result(
    raw: dict | None,
    candidate_sources: list[dict],
    completion_criteria: list[str] | None = None,
    answer_mode: str = "strict",
    valid_source_ids: set[int] | None = None,
    identifying_criteria: set[int] | list[int] | None = None,
    enforced_gaps: list[str] | None = None,
) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    completion_criteria = completion_criteria or []
    requirement_count = len(completion_criteria)
    source_count = len(candidate_sources)
    visible_source_ids = (
        set(range(1, source_count + 1))
        if valid_source_ids is None
        else set(valid_source_ids)
    )
    identifying = {
        int(index)
        for index in (identifying_criteria or ())
        if str(index).lstrip("-").isdigit() and 0 <= int(index) < requirement_count
    }
    ledger = []
    supported_source_ids: set[int] = set()
    supported_claims = 0
    dropped_gaps: list[str] = []
    for item in raw.get("ledger", []):
        if not isinstance(item, dict):
            continue
        source_ids = []
        for raw_id in item.get("source_ids", []):
            try:
                source_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if source_id in visible_source_ids and source_id not in source_ids:
                source_ids.append(source_id)
        support_status = (
            str(item.get("support_status") or item.get("support_level") or "missing")
            .strip()
            .lower()
        )
        if support_status not in {"supported", "partial", "missing", "rejected"}:
            support_status = "missing"
        support_level = str(item.get("support_level") or "missing").strip().lower()
        if support_level not in {"supported", "partial", "missing"}:
            support_level = (
                "missing" if support_status == "rejected" else support_status
            )
        if support_status == "supported":
            support_level = "supported"
        can_use = (
            bool(item.get("can_use_in_answer"))
            and support_level == "supported"
            and bool(source_ids)
        )
        if can_use:
            supported_source_ids.update(source_ids)
            supported_claims += 1
        claim_id = str(item.get("claim_id") or "").strip()[:80]
        if not claim_id:
            claim_id = f"claim_{len(ledger) + 1}"
        _, requirement_index = _parse_indexed_item(item, requirement_count)
        missing_text = str(item.get("missing") or "").strip()[:300]
        if missing_text and requirement_index is None:
            dropped_gaps.append(missing_text)
            missing_text = ""
        ledger.append(
            {
                "claim_id": claim_id,
                "requirement_index": requirement_index,
                "requirement": str(item.get("requirement") or "").strip()[:200],
                "proposed_claim": str(item.get("proposed_claim") or "").strip()[:500],
                "event_date": str(item.get("event_date") or "").strip()[:40],
                "publication_date": str(item.get("publication_date") or "").strip()[
                    :40
                ],
                "location": str(item.get("location") or "").strip()[:120],
                "source_ids": source_ids,
                "source_quality": str(item.get("source_quality") or "unknown")
                .strip()
                .lower()[:40],
                "support_status": support_status,
                "support_level": support_level,
                "can_use_in_answer": can_use,
                "missing": missing_text,
                "rejection_reason": str(item.get("rejection_reason") or "").strip()[
                    :300
                ],
            }
        )

    global_missing = []
    clue_gaps: list[str] = []
    recovered_gaps: list[str] = []
    for raw_gap in raw.get("global_missing", []):
        text, idx = _parse_indexed_item(raw_gap, requirement_count)
        if not text:
            continue
        if idx is None:
            if _recurring_gap(text, enforced_gaps):
                global_missing.append(text)
                recovered_gaps.append(text)
            else:
                dropped_gaps.append(text)
        elif idx in identifying:
            # "No source confirms the subject was born in 1886" is a true statement about
            # a clue the asker supplied. Blocking on it is the same category error as
            # demanding the clue be proven: it belongs in the disclosure, not in the gate.
            clue_gaps.append(text)
        else:
            global_missing.append(text)

    next_steps = []
    for step in raw.get("next_steps", []):
        step = str(step).strip()
        if step and ":" in step and step not in next_steps:
            next_steps.append(step[:500])

    # answer_ready is derived from validated structured fields only — the model's own
    # raw "answer_ready" self-report is not trusted, since a hallucinated (unindexed)
    # concern must not be able to block readiness just because the model believed it.
    indexed_rows = [item for item in ledger if item["requirement_index"] is not None]
    unconfirmed_clues: list[int] = []
    if answer_mode == "roundup":
        enough_roundup_evidence = (
            len(supported_source_ids) >= ROUNDUP_MIN_SOURCES
            or supported_claims >= ROUNDUP_MIN_CLAIMS
        )
        # A roundup claim can satisfy several broad requirements at once even though
        # the ledger schema stores only one requirement_index per row. Requiring an
        # indexed row for every criterion therefore creates false negatives. The
        # model's indexed global_missing list remains the deterministic blocker.
        answer_ready = bool(ledger) and not global_missing and enough_roundup_evidence
    else:
        supported_indices = {
            item["requirement_index"]
            for item in indexed_rows
            if item["support_level"] == "supported" and item["can_use_in_answer"]
        }
        # Criteria the user supplied as clues are searched for but not demanded as proof:
        # they single out the subject, and requiring a cited restatement of what the asker
        # already told us makes strict mode unreachable on the questions it exists for.
        # See requirements._identifying_indices. With none marked this is the whole set,
        # which is the gate exactly as it was.
        required_indices = set(range(requirement_count)) - identifying
        answer_ready = (
            bool(required_indices)
            and not global_missing
            and required_indices <= supported_indices
        )
        # A clue the run could not confirm is not a blocker, but the answer has to say so
        # rather than present an unverified identification as settled.
        unconfirmed_clues = sorted(identifying - supported_indices)
    admitted = [
        candidate_sources[source_id - 1] for source_id in sorted(supported_source_ids)
    ]
    return {
        "answer_ready": answer_ready,
        "unconfirmed_identifying": list(
            dict.fromkeys(
                [
                    completion_criteria[index]
                    for index in unconfirmed_clues
                    if 0 <= index < len(completion_criteria)
                ]
                + clue_gaps
            )
        ),
        "answer_mode": answer_mode,
        "ledger": ledger,
        "global_missing": list(dict.fromkeys(global_missing)),
        "dropped_gaps": list(dict.fromkeys(dropped_gaps)),
        "recovered_gaps": list(dict.fromkeys(recovered_gaps)),
        "next_steps": next_steps[:4],
        "reason": str(raw.get("reason") or "").strip()[:500],
        "candidate_count": source_count,
        "admissible_count": len(admitted),
        "supported_claim_count": supported_claims,
        "admissible_sources": admitted,
    }


def _normalize_evidence_challenge_result(
    raw: dict | None,
    ledger_result: dict | None,
    requirement_count: int = 0,
    identifying_criteria: set[int] | list[int] | None = None,
    enforced_gaps: list[str] | None = None,
) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    ledger_result = ledger_result if isinstance(ledger_result, dict) else {}
    identifying = {
        int(index)
        for index in (identifying_criteria or ())
        if str(index).lstrip("-").isdigit() and 0 <= int(index) < requirement_count
    }
    blocking_gaps = []
    clue_gaps = []
    dropped_gaps = []
    recovered_gaps: list[str] = []
    for raw_gap in raw.get("blocking_gaps", []):
        text, idx = _parse_indexed_item(raw_gap, requirement_count)
        if not text:
            continue
        if idx is None:
            if _recurring_gap(text, enforced_gaps):
                blocking_gaps.append(text)
                recovered_gaps.append(text)
            else:
                dropped_gaps.append(text)
        elif idx in identifying:
            # The challenge pass reads the same numbered criteria as the ledger, so it can
            # re-raise on a clue the ledger already set aside. Left blocking here, the gate
            # closes again one node later and the relaxation buys nothing.
            clue_gaps.append(text)
        else:
            blocking_gaps.append(text)
    next_steps = []
    for step in raw.get("next_steps", []):
        step = str(step).strip()
        if step and ":" in step and step not in next_steps:
            next_steps.append(step[:500])
    # answer_permitted is derived from the (already-sanitized) ledger readiness and the
    # (already-sanitized) blocking_gaps — the model's own raw "answer_permitted" flag is
    # not trusted, for the same reason as evidence_ledger's answer_ready above.
    answer_permitted = bool(ledger_result.get("answer_ready")) and not blocking_gaps
    return {
        "answer_permitted": answer_permitted,
        "blocking_gaps": list(dict.fromkeys(blocking_gaps)),
        "unconfirmed_identifying": list(dict.fromkeys(clue_gaps)),
        "dropped_gaps": list(dict.fromkeys(dropped_gaps)),
        "recovered_gaps": list(dict.fromkeys(recovered_gaps)),
        "next_steps": next_steps[:4],
        "reason": str(raw.get("reason") or "").strip()[:500],
    }
