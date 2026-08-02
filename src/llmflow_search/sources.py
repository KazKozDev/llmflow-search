"""Source extraction, normalization, and evidence auditing."""

import hashlib
import json
import re
from typing import Any

from .config import SOURCE_CONTENT_MAX_CHARS, TOTAL_SOURCES_MAX_CHARS
from .llm import _json_loads_best_effort


def _clip_text(text: str, limit: int = SOURCE_CONTENT_MAX_CHARS) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0].strip()


def _normalize_source_url(url: str) -> str:
    return (url or "").split("#", 1)[0].rstrip("/")


def _split_deep_search_context(context: str, source_lookup: dict[int, dict]) -> list[dict]:
    if not context:
        return []

    matches = list(re.finditer(r"(?m)^\[(\d+)\]\s+(.+)$", context))
    sources = []
    for pos, match in enumerate(matches):
        src_num = int(match.group(1))
        title = match.group(2).strip()
        content_start = match.end()
        content_end = matches[pos + 1].start() if pos + 1 < len(matches) else len(context)
        source_meta = source_lookup.get(src_num, {})
        content = _clip_text(context[content_start:content_end])
        url = source_meta.get("url", "")
        if content and url:
            sources.append(
                {
                    "title": source_meta.get("title") or title,
                    "url": url,
                    "published": source_meta.get("pub_date") or source_meta.get("published"),
                    "content": content,
                    "kind": "page",
                }
            )
    return sources


def _sources_from_tool_result(tool_name: str, tool_result: str) -> list[dict]:
    payload = _json_loads_best_effort(tool_result, {})
    if not isinstance(payload, dict):
        return []

    if tool_name == "web_read":
        text = _clip_text(payload.get("text", ""))
        url = payload.get("url", "")
        if payload.get("error") or not text or not url:
            return []
        source_quality = payload.get("source_type") if isinstance(payload.get("source_type"), dict) else {}
        candidate_links = [
            {"text": str(link.get("text", ""))[:200], "url": str(link.get("url", ""))}
            for link in payload.get("links", [])
            if isinstance(link, dict) and link.get("url")
        ]
        return [
            {
                "title": payload.get("title") or url,
                "url": url,
                "published": payload.get("pub_date"),
                "content": text,
                "kind": "page",
                "source_quality": source_quality,
                "is_listing_page": bool(payload.get("is_listing")),
                "candidate_links": candidate_links,
            }
        ]

    if tool_name == "web_deep_search":
        source_lookup = {}
        for source in payload.get("sources", []):
            if isinstance(source, dict) and source.get("num"):
                source_lookup[int(source["num"])] = source
        return _split_deep_search_context(payload.get("context", ""), source_lookup)

    if tool_name == "web_extract_tables":
        url = payload.get("url", "")
        tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []
        if payload.get("error") or not url or not tables:
            return []
        content = _clip_text(json.dumps({"tables": tables}, ensure_ascii=False, indent=2), 12000)
        return [
            {
                "title": f"Structured tables from {url}",
                "url": url,
                "published": payload.get("published"),
                "content": content,
                "kind": "table",
            }
        ]

    if tool_name == "web_parse_file":
        url = payload.get("url", "")
        if payload.get("error") or not url:
            return []
        evidence_payload = {
            "file_type": payload.get("file_type"),
            "tables": payload.get("tables", []),
            "pages": payload.get("pages", []),
            "json": payload.get("json"),
        }
        content = _clip_text(json.dumps(evidence_payload, ensure_ascii=False, indent=2), 12000)
        if not content:
            return []
        return [
            {
                "title": f"Parsed file from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "file",
            }
        ]

    if tool_name == "web_fetch_json":
        url = payload.get("url", "")
        if payload.get("error") or not url or "json" not in payload:
            return []
        content = _clip_text(json.dumps({"json": payload.get("json")}, ensure_ascii=False, indent=2), 12000)
        if not content:
            return []
        return [
            {
                "title": f"JSON API response from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "json",
            }
        ]

    if tool_name == "tool_code_run_sandboxed":
        if not payload.get("ok"):
            return []
        _result_raw = payload.get("result")
        result: dict = _result_raw if isinstance(_result_raw, dict) else {}
        _rows_raw = result.get("rows")
        rows: list = _rows_raw if isinstance(_rows_raw, list) else []
        if not rows:
            return []
        url = ""
        for row in rows:
            if isinstance(row, dict) and row.get("source_url"):
                url = row["source_url"]
                break
        url = url or "recipe://sandboxed-extraction"
        content = _clip_text(json.dumps(result, ensure_ascii=False, indent=2), 12000)
        return [
            {
                "title": f"Sandboxed extraction recipe result from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "recipe_rows",
            }
        ]

    if tool_name == "browser_extract_tables":
        url = payload.get("url", "")
        tables = payload.get("tables") if isinstance(payload.get("tables"), list) else []
        if not url or not tables:
            return []
        content = _clip_text(json.dumps({"tables": tables}, ensure_ascii=False, indent=2), 12000)
        return [
            {
                "title": payload.get("title") or f"Browser tables from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "browser_table",
            }
        ]

    return []


def generic_sources_from_tool_result(tool_name: str, tool_result: str) -> list[dict]:
    """Best-effort source for an arbitrary (non-footnote) MCP tool.

    We know nothing about the tool's schema, so we treat its textual output as one
    evidence item. A stable pseudo-URL (`tool://name/<hash>`) gives it the unique,
    non-empty key that `_merge_sources` dedups on — identical outputs collapse, distinct
    ones are kept — so tool results count as grounded sources in generic mode.
    """
    text = _clip_text(tool_result or "")
    if not text.strip():
        return []
    digest = hashlib.md5(f"{tool_name}\n{text}".encode()).hexdigest()[:12]
    return [
        {
            "title": tool_name,
            "url": f"tool://{tool_name}/{digest}",
            "published": None,
            "content": text,
            "kind": "tool_output",
        }
    ]


def _merge_sources(existing: list[dict], new_sources: list[dict]) -> list[dict]:
    merged = list(existing or [])
    seen = {_normalize_source_url(src.get("url", "")) for src in merged}
    for source in new_sources:
        key = _normalize_source_url(source.get("url", ""))
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(source)
    return merged


def _format_sources_for_llm(sources: list[dict]) -> tuple[str, set[int]]:
    parts = []
    valid_ids = set()
    total_chars = 0

    for src_id, source in enumerate(sources or [], 1):
        content = _clip_text(source.get("content", ""))
        if not content:
            continue
        block = (
            f"[{src_id}]\n"
            f"Title: {source.get('title') or 'Untitled'}\n"
            f"URL: {source.get('url') or ''}\n"
            f"Published: {source.get('published') or 'unknown'}\n"
            f"Source type: {(source.get('source_quality') or {}).get('source_type') or source.get('kind') or 'unknown'}\n"
            f"Content:\n{content}\n"
        )
        if total_chars + len(block) > TOTAL_SOURCES_MAX_CHARS and parts:
            break
        total_chars += len(block)
        valid_ids.add(src_id)
        parts.append(block)

    return "\n".join(parts), valid_ids


def _effective_question(task: str) -> str:
    marker = "\n\nNew question:"
    if marker in task:
        return task.rsplit(marker, 1)[1].strip()
    return task.strip()


def _structured_source_count(sources: list[dict]) -> int:
    structured_kinds = {"table", "file", "json", "recipe_rows", "browser_table"}
    return sum(1 for source in sources or [] if isinstance(source, dict) and source.get("kind") in structured_kinds)


def _audit_evidence_state(state: dict[str, Any]) -> dict:
    ledger_result = state.get("evidence_ledger_result")
    if isinstance(ledger_result, dict):
        admitted = state.get("admissible_sources", []) or state.get("sources", []) or []
        sources = [source for source in admitted if isinstance(source, dict)]
        candidate_count = int(ledger_result.get("candidate_count") or len(state.get("candidate_sources", []) or []))
    else:
        sources = [source for source in state.get("sources", []) or [] if isinstance(source, dict)]
        candidate_count = len(sources)
    gaps = []
    strategy_hints = []

    if not sources:
        if candidate_count:
            gaps.append("No fetched sources support the evidence ledger.")
            strategy_hints.append("Use tools to close evidence-ledger gaps.")
        else:
            gaps.append("No fetched evidence sources are available.")
            strategy_hints.append("Fetch source pages before answering.")

    passed = not gaps
    return {
        "passed": passed,
        "gaps": list(dict.fromkeys(gaps)),
        "strategy_hints": list(dict.fromkeys(strategy_hints)),
        "source_count": len(sources),
        "candidate_source_count": candidate_count,
        "structured_source_count": _structured_source_count(sources),
    }


def _payload_row_count(payload: dict) -> int:
    if isinstance(payload.get("tables"), list):
        return sum(len(table.get("rows", []) or []) for table in payload["tables"] if isinstance(table, dict))
    if isinstance(payload.get("json"), dict | list):
        raw = payload["json"]
        if isinstance(raw, list):
            return len(raw)
        for key in ("rows", "data", "prices", "items", "results"):
            value = raw.get(key) if isinstance(raw, dict) else None
            if isinstance(value, list):
                return len(value)
    _res_raw = payload.get("result")
    _result: dict = _res_raw if isinstance(_res_raw, dict) else {}
    _rows_raw = _result.get("rows")
    _rows: list = _rows_raw if isinstance(_rows_raw, list) else []
    return len(_rows)
