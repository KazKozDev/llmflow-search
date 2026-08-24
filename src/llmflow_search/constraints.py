"""What the question actually asks for, condition by condition, and what closed each one.

A benchmark question is a conjunction: born in 1886, travelled April to November, was
mistaken for a shaman. Nothing in the run held the idea of *a condition with a status*.
There were thirty list-shaped keys in search memory and a scratchpad, and what reached
the prompt was ``scratchpad[-2500:]`` — so run memory was lost precisely in the middle,
exactly when a question had taken enough steps for the middle to matter. A round could
therefore go looking again for something an earlier round had already established, and
nothing in the state could contradict it.

The registry is a projection, not a second store. Conditions are the numbered
``completion_criteria`` the requirements node already produces, and their statuses are
read off the evidence-ledger rows that already index against those same numbers. Nothing
here can drift from the ledger, because there is nothing here to drift: it is the ledger,
read by condition instead of by claim. It is small enough to send whole.
"""

from dataclasses import dataclass, field

from .sources import _normalize_source_url

# Strongest first. A condition with both a supporting and a rejecting row is reported as
# settled rather than open — the disagreement is visible in the rows themselves, and
# treating it as open sends the run searching for something it has already decided twice.
_STATUS_PRECEDENCE = ("satisfied", "refuted", "partial", "open")

_STATUS_MARK = {
    "satisfied": "OK  ",
    "refuted": "NO  ",
    "partial": "PART",
    "open": "??  ",
}


@dataclass
class Constraint:
    """One condition from the question, and what settled it."""

    index: int
    text: str
    status: str = "open"
    source_urls: list[str] = field(default_factory=list)
    # The claim the ledger recorded for this condition — what the run believes it read.
    quote: str = ""

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "text": self.text,
            "status": self.status,
            "source_urls": list(self.source_urls),
            "quote": self.quote,
        }


def _row_status(row: dict) -> str:
    status = str(row.get("support_status") or "").strip().lower()
    if status == "rejected":
        return "refuted"
    if status == "supported" and row.get("can_use_in_answer"):
        return "satisfied"
    if status in ("supported", "partial"):
        return "partial"
    return "open"


def build_registry(
    completion_criteria: list[str],
    ledger_result: dict | None,
    candidate_sources: list[dict] | None = None,
) -> list[dict]:
    """Project the ledger onto the question's conditions."""
    registry = [
        Constraint(index=index, text=str(text))
        for index, text in enumerate(completion_criteria or [])
    ]
    if not registry:
        return []

    sources = list(candidate_sources or [])
    ledger_result = ledger_result if isinstance(ledger_result, dict) else {}
    for row in ledger_result.get("ledger", []) or []:
        if not isinstance(row, dict):
            continue
        index = row.get("requirement_index")
        if not isinstance(index, int) or not (0 <= index < len(registry)):
            continue
        constraint = registry[index]
        status = _row_status(row)
        if _STATUS_PRECEDENCE.index(status) < _STATUS_PRECEDENCE.index(
            constraint.status
        ):
            constraint.status = status
            constraint.quote = str(row.get("proposed_claim") or "")[:300]
        for source_id in row.get("source_ids", []) or []:
            if not isinstance(source_id, int) or not (1 <= source_id <= len(sources)):
                continue
            source = sources[source_id - 1]
            url = _normalize_source_url(
                str(source.get("url") or "") if isinstance(source, dict) else ""
            )
            if url and url not in constraint.source_urls:
                constraint.source_urls.append(url)

    return [constraint.as_dict() for constraint in registry]


def open_constraints(registry: list[dict] | None) -> list[dict]:
    """Conditions nothing has settled yet — the only ones more searching can help."""
    return [
        item
        for item in registry or []
        if isinstance(item, dict) and item.get("status") in ("open", "partial")
    ]


def all_settled(registry: list[dict] | None) -> bool:
    return bool(registry) and not open_constraints(registry)


def format_for_prompt(registry: list[dict] | None, max_quote_chars: int = 160) -> str:
    """The whole registry, compactly. Sent in full because it fits in full.

    The scratchpad tail cannot say which conditions are already closed — it says what the
    last few steps printed. This says what the run knows, and by which page it knows it.
    """
    if not registry:
        return "(no conditions extracted)"
    lines = []
    for item in registry:
        if not isinstance(item, dict):
            continue
        mark = _STATUS_MARK.get(str(item.get("status")), "??  ")
        line = f"  [{mark}] {item.get('index')}. {item.get('text')}"
        quote = str(item.get("quote") or "").strip()
        if quote:
            line += f"\n         claim: {quote[:max_quote_chars]}"
        urls = item.get("source_urls") or []
        if urls:
            line += f"\n         from:  {', '.join(str(url) for url in urls[:3])}"
        lines.append(line)
    return "\n".join(lines)
