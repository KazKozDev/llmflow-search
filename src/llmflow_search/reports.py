"""Run assimilation into memory and optional JSON debug reports."""

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

from . import attribution
from . import memory as _memmod
from .config import PDF_LOGO_PATH, PDF_REPORTS_DIR
from .memory import _slug_key
from .pdf_report import render_report_pdf, report_pdf_filename
from .search_memory import _append_unique, _merge_search_memory


def _assimilate_research(state: dict) -> None:
    """Record research experience regardless of outcome — success and failure both teach."""
    verification = state.get("verification_result", {})
    succeeded = (
        verification.get("task_complete") is True
        and verification.get("insufficient_evidence") is not True
    )

    memory = _merge_search_memory(state.get("search_memory"))
    store = _memmod._get_research_store()
    requirements = state.get("requirements_result", {})
    current_strategy = (
        memory.get("current_strategy") or "source-grounded iterative search"
    )

    # Record strategy outcome
    meta = {
        "query_count": len(memory.get("attempted_queries", [])),
        "source_count": len(state.get("sources", [])),
        "requirements": requirements,
        "gaps_resolved": verification.get("gaps", []) if succeeded else [],
    }
    store.record_strategy(current_strategy, success=succeeded, won=succeeded, meta=meta)
    for candidate in memory.get("strategy_candidates", []):
        desc = candidate.get("desc", "")
        if desc and desc != current_strategy:
            store.record_strategy(
                desc, success=False, won=False, meta={"reason": "not selected"}
            )

    # Collect source domains (useful) and read-but-skipped domains (not useful)
    source_domains: list[str] = []
    for source in state.get("sources", []):
        url = source.get("url", "")
        m = re.search(r"https?://([^/]+)", url)
        if m:
            _append_unique(source_domains, m.group(1).lower())

    read_urls = memory.get("read_urls", [])
    barren_domains: list[str] = []
    for url in read_urls:
        m = re.search(r"https?://([^/]+)", url)
        if m:
            domain = m.group(1).lower()
            if domain not in source_domains:
                _append_unique(barren_domains, domain)

    if succeeded:
        skill = {
            "name": f"research-{_slug_key(str(requirements.get('target') or state.get('task', 'task')))}",
            "trigger": requirements.get("target") or state.get("task", ""),
            "steps": [
                f"Use strategy: {current_strategy}",
                "Search for source pages, then fetch pages before answering.",
                "Verify both source grounding and task completion gaps before finishing.",
            ],
            "source_domains": source_domains[:10],
            "barren_domains": barren_domains[:10],
            "success_rate": 1.0,
            "use_count": 0,
        }
        store.save_skill(skill)

    store.add_experience(
        {
            "task": state.get("task", ""),
            "result": "success" if succeeded else "failure",
            "strategy": current_strategy,
            "requirements": requirements,
            "queries": memory.get("attempted_queries", []),
            "sources": [s.get("url", "") for s in state.get("sources", [])],
            "barren_domains": barren_domains[:10],
        }
    )


def _build_debug_report(state: dict) -> dict:
    """Use footnote-mcp's richer report builder when it's installed; otherwise fall back to a
    small local report so the agent stays usable against any MCP server."""
    try:
        from footnote_mcp.tools_data import (  # pyright: ignore[reportMissingImports]
            build_research_debug_report,
        )
    except ImportError:
        search_memory = state.get("search_memory", {}) or {}
        verification = state.get("verification_result", {}) or {}
        return {
            "task": state.get("task", ""),
            "requirements": state.get("requirements_result", {}),
            "attempted_queries": search_memory.get("attempted_queries", []),
            "sources": [
                {"url": s.get("url"), "title": s.get("title"), "kind": s.get("kind")}
                for s in state.get("sources", [])
            ],
            "verification": verification,
            "gaps": verification.get("gaps", []),
        }
    return build_research_debug_report(
        task=state.get("task", ""),
        requirements=state.get("requirements_result", {}),
        search_memory=state.get("search_memory", {}),
        sources=state.get("sources", []),
        verification=state.get("verification_result", {}),
    )


def _write_debug_report(state: dict) -> str | None:
    if os.getenv("LLMFLOW_SEARCH_DEBUG_REPORTS", "0") != "1":
        return None
    output_dir = Path(
        os.getenv("LLMFLOW_SEARCH_DEBUG_REPORT_DIR", "~/.llmflow-search/debug_reports")
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = _build_debug_report(state)
    path = (
        output_dir
        / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{_slug_key(state.get('task', 'task'))}.json"
    )
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return str(path)


_ISO_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _source_date(source: dict) -> str:
    """The source's publication date as ISO, or "" when it carries none.

    Only a full ISO date is accepted, and only one the calendar admits. The shape is not
    enough: "published" comes from whatever the page declared, so "2026-02-30" reaches
    here looking exactly like a date, and a window built on it took a finished run down
    with a ValueError while writing its report. A half-parsed date is worse than no date
    anyway — the window is the reader's only check on whether figures from different
    months were mixed, and a window built from guesses cannot be checked.
    """
    match = _ISO_DATE.search(str(source.get("published") or ""))
    if not match:
        return ""
    try:
        datetime.strptime(match.group(1), "%Y-%m-%d")
    except ValueError:
        return ""
    return match.group(1)


def _source_type(source: dict) -> str:
    return str(
        (source.get("source_quality") or {}).get("source_type")
        or source.get("kind")
        or ""
    ).strip()


# How far back a source can sit from the newest one and still belong to the same "now".
# Beyond it the source is background: still evidence, but not part of the current picture.
_FRESH_WINDOW_DAYS = 14


def _evidence_window(sources: list[dict]) -> str:
    """One line stating what the evidence spans — and naming what does not belong to now.

    A plain first-to-last range hides exactly the mixing it was added to expose. One
    February page cited beside today's news prints as "2026-02-19 to 2026-09-20", a
    seven-month window that describes neither: the reader sees a range and assumes an
    evenly covered period, when in fact the report is today's news plus one stale page.

    So the recent sources set the window and the older ones are counted and dated
    separately. The split is stated, not averaged away, which is the only form in which a
    reader can act on it.
    """
    dates = sorted(date for date in (_source_date(s) for s in sources) if date)
    undated = len(sources) - len(dates)
    if not dates:
        parts = ["Evidence window: no source carries a publication date"]
    else:
        newest = datetime.strptime(dates[-1], "%Y-%m-%d")
        cutoff = newest - timedelta(days=_FRESH_WINDOW_DAYS)
        fresh = [d for d in dates if datetime.strptime(d, "%Y-%m-%d") >= cutoff]
        older = [d for d in dates if d not in fresh]
        span = fresh[0] if fresh[0] == fresh[-1] else f"{fresh[0]} to {fresh[-1]}"
        if older:
            oldest = older[0] if older[0] == older[-1] else f"{older[0]} to {older[-1]}"
            parts = [
                f"Evidence window: {span} ({len(fresh)} of {len(dates)} dated sources)",
                f"{len(older)} older, {oldest}",
            ]
        else:
            parts = [f"Evidence window: {span}"]
    parts.append(f"{len(sources)} source{'' if len(sources) == 1 else 's'}")
    if undated:
        parts.append(f"{undated} undated")
    parts.append(f"compiled {datetime.now().strftime('%Y-%m-%d')}")
    return " · ".join(parts)


def _cited_sources(answer: str, sources: list[dict]) -> list[tuple[int, dict]]:
    """The sources the finished text actually points at, with the numbers it uses.

    The admitted list is not the evidence behind a report. Two filters sit between them.
    The prompt has a total size, so a long list is cut off partway and the writer is never
    shown the tail (``sources._format_sources_for_llm``); and of what it is shown, it cites
    what it used. Printing the whole admitted list as the bibliography therefore credits
    the report with sources that could not have informed a word of it — measured on a
    60-source run, 47 reached the writer and the report claimed all 60.

    The stated evidence window is the worse half of that. It exists so a reader can check
    whether figures from different months were mixed, and computing it over pages the text
    never used means the one line offered for checking is itself unchecked. A source that
    is never cited cannot widen the window of a report it is not in.

    The numbers are kept as they are, gaps and all: they are the markers in the text, and
    renumbering the bibliography would point every citation at the wrong entry.
    """
    cited = set(attribution.citations_in(answer))
    pairs = [
        (index, source)
        for index, source in enumerate(sources, 1)
        if index in cited and isinstance(source, dict)
    ]
    # A report with no markers at all is a refusal or a failure of the citation pass, not
    # a report with no evidence. Falling back to the admitted list keeps the bibliography
    # from vanishing in exactly the case someone needs to look at it.
    return pairs or list(enumerate(sources, 1))


def _report_markdown(state: dict) -> str:
    """Combine the verified answer and its admitted sources into one PDF-ready report."""
    task = str(state.get("task") or "Research report").strip()
    answer = str(state.get("final_answer") or "").strip()
    cited = _cited_sources(answer, state.get("sources", []) or [])
    lines = [f"# {task}", ""]
    if cited:
        lines.extend([f"*{_evidence_window([source for _, source in cited])}*", ""])
    lines.append(answer)
    if cited:
        # Date and type per source, not just a link: they are what lets a reader check
        # the report's cutoff and tell a vendor's claim from an independent account.
        lines.extend(["", "## Sources", ""])
        for index, source in cited:
            title = str(
                source.get("title") or source.get("url") or f"Source {index}"
            ).strip()
            url = str(source.get("url") or "").strip()
            entry = f"{index}. [{title}]({url}) - {url}" if url else f"{index}. {title}"
            meta = [
                part
                for part in (_source_date(source) or "undated", _source_type(source))
                if part
            ]
            lines.append(f"{entry} ({', '.join(meta)})" if meta else entry)
    return "\n".join(lines).strip() + "\n"


def _write_pdf_report(state: dict) -> str | None:
    """Write a PDF only for a successfully verified, source-grounded answer."""
    verification = state.get("verification_result", {}) or {}
    if verification.get("task_complete") is not True:
        return None
    if verification.get("insufficient_evidence") is True:
        return None
    if not state.get("sources") or not str(state.get("final_answer") or "").strip():
        return None

    task = str(state.get("task") or "Research report").strip()
    research_id = f"res_{uuid4().hex[:12]}"
    output_path = Path(PDF_REPORTS_DIR).expanduser() / report_pdf_filename(
        research_id, task
    )
    logo_path = Path(PDF_LOGO_PATH).expanduser() if PDF_LOGO_PATH else None
    render_report_pdf(
        research_id=research_id,
        query=task,
        report_markdown=_report_markdown(state),
        output_path=output_path,
        logo_path=logo_path,
    )
    return str(output_path)
