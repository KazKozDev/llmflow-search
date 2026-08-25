"""Deterministic evidence-ledger schemas and normalization contracts."""

from .config import ROUNDUP_MIN_CLAIMS, ROUNDUP_MIN_SOURCES


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
    for raw_gap in raw.get("global_missing", []):
        text, idx = _parse_indexed_item(raw_gap, requirement_count)
        if not text:
            continue
        if idx is None:
            dropped_gaps.append(text)
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
        required_indices = set(range(requirement_count))
        answer_ready = (
            bool(required_indices)
            and not global_missing
            and supported_indices == required_indices
        )
    admitted = [
        candidate_sources[source_id - 1] for source_id in sorted(supported_source_ids)
    ]
    return {
        "answer_ready": answer_ready,
        "answer_mode": answer_mode,
        "ledger": ledger,
        "global_missing": list(dict.fromkeys(global_missing)),
        "dropped_gaps": list(dict.fromkeys(dropped_gaps)),
        "next_steps": next_steps[:4],
        "reason": str(raw.get("reason") or "").strip()[:500],
        "candidate_count": source_count,
        "admissible_count": len(admitted),
        "supported_claim_count": supported_claims,
        "admissible_sources": admitted,
    }


def _normalize_evidence_challenge_result(
    raw: dict | None, ledger_result: dict | None, requirement_count: int = 0
) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    ledger_result = ledger_result if isinstance(ledger_result, dict) else {}
    blocking_gaps = []
    dropped_gaps = []
    for raw_gap in raw.get("blocking_gaps", []):
        text, idx = _parse_indexed_item(raw_gap, requirement_count)
        if not text:
            continue
        if idx is None:
            dropped_gaps.append(text)
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
        "dropped_gaps": list(dict.fromkeys(dropped_gaps)),
        "next_steps": next_steps[:4],
        "reason": str(raw.get("reason") or "").strip()[:500],
    }
