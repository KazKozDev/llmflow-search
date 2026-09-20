"""The sidecar a run leaves behind so its answers can be re-scored without re-running it.

The JSONL trace records what the run *did*. Scoring what it *said* needs two more things
the trace deliberately does not carry: the answer text and the full text of every source
a citation can point at. Without them, measuring attribution means re-running the
pipeline — twenty minutes and a fresh set of search results — every time the judge, the
prompt or the threshold changes.

One record per answered question, written next to the trace. Scoring is then an offline
pass over a file, which is what makes a before/after comparison mean anything: both sides
are re-scored by the same judge from the same stored evidence.
"""

import json
from pathlib import Path
from typing import Any

from .sources import _clip_text, _format_sources_for_llm

# Full source text is what "supported by the cited source" has to be checked against, but
# a corpus page can run to megabytes. This is the same ceiling the pipeline puts on the
# text it will consider for one source, so nothing is dropped that the run could have used.
SOURCE_TEXT_MAX_CHARS = 25000


def source_records(sources: list[dict], question: str) -> list[dict]:
    """Every source a ``[n]`` marker can refer to, numbered exactly as the prompt was.

    ``shown`` is re-derived rather than captured: ``_format_sources_for_llm`` is
    deterministic in its inputs, and re-deriving it here keeps the record honest about
    which excerpt the answer model actually had in front of it — the difference between
    "the page does not say this" and "the page says it outside the excerpt we showed".
    """
    shown_text, _valid = _format_sources_for_llm(sources or [], question)
    shown_by_id = _split_shown_blocks(shown_text)
    records = []
    for source_id, source in enumerate(sources or [], 1):
        records.append(
            {
                "id": source_id,
                "url": source.get("url") or "",
                "title": source.get("title") or "",
                "kind": source.get("kind") or "",
                "content": _clip_text(
                    source.get("content", "") or "", SOURCE_TEXT_MAX_CHARS
                ),
                "shown": shown_by_id.get(source_id, ""),
            }
        )
    return records


def _split_shown_blocks(shown_text: str) -> dict[int, str]:
    """Recover the per-source excerpt from the single blob the prompt receives."""
    blocks: dict[int, str] = {}
    current: int | None = None
    buffer: list[str] = []
    for line in (shown_text or "").splitlines():
        stripped = line.strip()
        if (
            stripped.startswith("[")
            and stripped.endswith("]")
            and stripped[1:-1].isdigit()
        ):
            if current is not None:
                blocks[current] = "\n".join(buffer).strip()
            current, buffer = int(stripped[1:-1]), []
            continue
        buffer.append(line)
    if current is not None:
        blocks[current] = "\n".join(buffer).strip()
    return blocks


def write_record(path: str | Path, record: dict[str, Any]) -> None:
    """Append one answered question. Never raises — this is a measurement side channel."""
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def read_records(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def sidecar_path(trace_path: str | Path) -> Path:
    """The answers file that belongs to a given trace."""
    trace_file = Path(trace_path)
    stem = trace_file.name
    for suffix in (".trace.jsonl", ".jsonl"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return trace_file.with_name(stem + ".answers.jsonl")
