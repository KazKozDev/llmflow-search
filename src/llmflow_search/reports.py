"""Run assimilation into memory and optional JSON debug reports."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

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


def _report_markdown(state: dict) -> str:
    """Combine the verified answer and its admitted sources into one PDF-ready report."""
    task = str(state.get("task") or "Research report").strip()
    answer = str(state.get("final_answer") or "").strip()
    lines = [f"# {task}", "", answer]
    sources = state.get("sources", []) or []
    if sources:
        lines.extend(["", "## Sources", ""])
        for index, source in enumerate(sources, 1):
            title = str(
                source.get("title") or source.get("url") or f"Source {index}"
            ).strip()
            url = str(source.get("url") or "").strip()
            lines.append(
                f"{index}. [{title}]({url}) - {url}" if url else f"{index}. {title}"
            )
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
