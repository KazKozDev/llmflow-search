"""Source extraction, normalization, and evidence auditing."""

import hashlib
import json
import re
from typing import Any

from .config import (
    PASSAGE_RELEVANCE_EXCERPTS,
    SOURCE_CONTENT_MAX_CHARS,
    TOTAL_SOURCES_MAX_CHARS,
)
from .llm import _json_loads_best_effort
from .passages import relevant_excerpt


def _clip_text(text: str, limit: int = SOURCE_CONTENT_MAX_CHARS) -> str:
    text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0].strip()


def _normalize_source_url(url: str) -> str:
    return (url or "").split("#", 1)[0].rstrip("/")


_URL_IN_TEXT = re.compile(r"https?://[^\s<>\"'\]\)]+")


def _urls_in_text(text: str) -> set[str]:
    """Every address written in a piece of text, normalized for comparison."""
    return {
        normalized
        for raw in _URL_IN_TEXT.findall(text or "")
        if (normalized := _normalize_source_url(raw.rstrip(".,;:")))
    }


def _urls_in_sources(sources: list[dict] | None) -> set[str]:
    """Addresses the agent has actually seen: each source's own URL and any inside its text."""
    found: set[str] = set()
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        own = _normalize_source_url(str(source.get("url") or ""))
        if own:
            found.add(own)
        found |= _urls_in_text(str(source.get("content") or ""))
        for link in source.get("candidate_links") or []:
            if isinstance(link, dict):
                linked = _normalize_source_url(str(link.get("url") or ""))
                if linked:
                    found.add(linked)
    return found


def _split_deep_search_context(
    context: str, source_lookup: dict[int, dict]
) -> list[dict]:
    if not context:
        return []

    matches = list(re.finditer(r"(?m)^\[(\d+)\]\s+(.+)$", context))
    sources = []
    for pos, match in enumerate(matches):
        src_num = int(match.group(1))
        title = match.group(2).strip()
        content_start = match.end()
        content_end = (
            matches[pos + 1].start() if pos + 1 < len(matches) else len(context)
        )
        source_meta = source_lookup.get(src_num, {})
        content = _clip_text(context[content_start:content_end])
        url = source_meta.get("url", "")
        if content and url and not _looks_like_block_page(content):
            sources.append(
                {
                    "title": source_meta.get("title") or title,
                    "url": url,
                    "published": source_meta.get("pub_date")
                    or source_meta.get("published"),
                    "content": content,
                    "kind": "page",
                }
            )
    return sources


# Normalized search records (papers, encyclopedia entries, repositories, archived
# captures) all share the {title, url, snippet/text, published, authors} shape the
# server returns, so one branch covers every one of them.
_SEARCH_RECORD_KINDS = {
    "papers_search": "paper",
    "encyclopedia_search": "encyclopedia",
    "github_search": "repository",
    "archive_search": "archive",
}

# A record earns a source slot only when its own payload carries substantive text —
# a paper abstract, a repository description, an archived snapshot body. Anything
# shorter is a discovery pointer, and pointers are not evidence.
_MIN_RECORD_CONTENT_CHARS = 80


# What an interstitial says instead of the page. Text like this is not evidence: it
# supports nothing, yet as a source it makes the run look like it has one — the
# controller then reports DONE, the ledger finds nothing, and the round repeats.
_BLOCK_PAGE_MARKERS = (
    "just a moment",
    "verify you are human",
    "are you a robot",
    "checking your browser",
    "enable javascript and cookies to continue",
    "cf-chl",
    "recaptcha",
    "hcaptcha",
    "captcha",
    "access denied",
    "request blocked",
    "ddos protection",
)
# Only a short page can be misread this way: a long article that happens to mention
# captchas is a real article, and dropping it would lose a legitimate source.
_BLOCK_PAGE_MAX_CHARS = 1500


def _looks_like_block_page(text: str) -> bool:
    body = (text or "").strip()
    if not body or len(body) > _BLOCK_PAGE_MAX_CHARS:
        return False
    lowered = body.lower()
    return any(marker in lowered for marker in _BLOCK_PAGE_MARKERS)


def _sources_from_tool_result(
    tool_name: str, tool_result: str, context: dict | None = None
) -> list[dict]:
    payload = _json_loads_best_effort(tool_result, {})
    if not isinstance(payload, dict):
        return []

    if tool_name == "web_read":
        text = _clip_text(payload.get("text", ""))
        url = payload.get("url", "")
        if payload.get("error") or not text or not url:
            return []
        if _looks_like_block_page(text):
            return []
        source_quality = (
            payload.get("source_type")
            if isinstance(payload.get("source_type"), dict)
            else {}
        )
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
        tables = (
            payload.get("tables") if isinstance(payload.get("tables"), list) else []
        )
        if payload.get("error") or not url or not tables:
            return []
        content = _clip_text(
            json.dumps({"tables": tables}, ensure_ascii=False, indent=2), 12000
        )
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
        content = _clip_text(
            json.dumps(evidence_payload, ensure_ascii=False, indent=2), 12000
        )
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
        content = _clip_text(
            json.dumps({"json": payload.get("json")}, ensure_ascii=False, indent=2),
            12000,
        )
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

    if tool_name in ("browser_extract_tables", "browser_extract_tables_for_date_range"):
        url = payload.get("url", "")
        tables = (
            payload.get("tables") if isinstance(payload.get("tables"), list) else []
        )
        if not url or not tables:
            return []
        # The date-range variant returns the same table payload plus the range it applied.
        # Keeping the range in the content is what makes the rows verifiable.
        table_payload: dict = {"tables": tables}
        if payload.get("date_range"):
            table_payload = {"date_range": payload["date_range"], **table_payload}
        content = _clip_text(
            json.dumps(table_payload, ensure_ascii=False, indent=2), 12000
        )
        return [
            {
                "title": payload.get("title") or f"Browser tables from {url}",
                "url": url,
                "published": None,
                "content": content,
                "kind": "browser_table",
            }
        ]

    if tool_name in _SEARCH_RECORD_KINDS:
        kind = _SEARCH_RECORD_KINDS[tool_name]
        records = []
        for item in payload.get("results") or []:
            if not isinstance(item, dict):
                continue
            url = item.get("url") or ""
            body = str(item.get("text") or "")
            content = _clip_text(body or str(item.get("snippet") or ""))
            if not url or len(content) < _MIN_RECORD_CONTENT_CHARS:
                continue
            record_source = item.get("source") or tool_name
            authors = [str(author) for author in (item.get("authors") or []) if author]
            record = {
                "title": item.get("title") or url,
                "url": url,
                "published": item.get("published"),
                "content": content,
                "kind": kind,
                "record_source": record_source,
                # An abstract is what the index holds, not the document — said in the
                # source's type line rather than prepended to its text. Inside the
                # content it read as a warning about the evidence as a whole: with four
                # index records beside one fetched page, the ledger concluded it held
                # "only abstracts" and refused a page it had already read in full.
                "source_quality": {
                    "source_type": f"{record_source} {kind} record"
                    + ("" if body else " (abstract only)")
                },
            }
            if authors:
                record["authors"] = authors
            if item.get("identifiers"):
                record["identifiers"] = item["identifiers"]
            records.append(record)
        return records

    if tool_name == "web_archive_fetch":
        text = _clip_text(payload.get("text", ""))
        snapshot_url = payload.get("snapshot_url") or ""
        if (
            payload.get("error")
            or payload.get("fetch_error")
            or not text
            or not snapshot_url
        ):
            return []
        original_url = payload.get("url") or snapshot_url
        return [
            {
                "title": payload.get("title") or f"Archived snapshot of {original_url}",
                "url": snapshot_url,
                "published": payload.get("published"),
                "content": text,
                "kind": "archive",
                "original_url": original_url,
            }
        ]

    if tool_name == "web_crawl":
        crawled = []
        for page in payload.get("pages") or []:
            if not isinstance(page, dict) or page.get("error"):
                continue
            url = page.get("url") or ""
            text = _clip_text(page.get("text", ""))
            if not url or not text or _looks_like_block_page(text):
                continue
            crawled.append(
                {
                    "title": page.get("title") or url,
                    "url": url,
                    "published": page.get("published"),
                    "content": text,
                    "kind": "page",
                }
            )
        return crawled

    if tool_name == "web_extract":
        # The browser session holds the address, not the tool result, so provenance
        # comes from the caller's context. Without it the text cannot be cited.
        url = _normalize_source_url(str((context or {}).get("browser_url") or ""))
        text = _clip_text(payload.get("text") or "")
        if not text and isinstance(payload.get("elements"), dict):
            text = _clip_text(
                "\n".join(
                    f"{ref}: {value}"
                    for ref, value in payload["elements"].items()
                    if value
                )
            )
        if not url or not text:
            return []
        return [
            {
                "title": f"Browser page text from {url}",
                "url": url,
                "published": None,
                "content": text,
                "kind": "page",
            }
        ]

    return []


def generic_sources_from_tool_result(
    tool_name: str, tool_result: str, context: dict | None = None
) -> list[dict]:
    """Best-effort source for an arbitrary (non-footnote) MCP tool.

    We know nothing about the tool's schema, so we treat its textual output as one
    evidence item. A stable pseudo-URL (`tool://name/<hash>`) gives it the unique,
    non-empty key that `_merge_sources` dedups on — identical outputs collapse, distinct
    ones are kept — so tool results count as grounded sources in generic mode.
    """
    del context  # no footnote-specific provenance to resolve for an unknown tool
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


def _source_worth(source: dict) -> tuple[int, int]:
    """Ranking used when two sources describe the same URL: fetched body beats record."""
    fetched = 1 if source.get("kind") in _FETCHED_KINDS else 0
    return fetched, len(str(source.get("content") or ""))


def _merge_sources(existing: list[dict], new_sources: list[dict]) -> list[dict]:
    """Merge by URL, keeping the richest version rather than the first one seen.

    An index record and the page it points at share a URL. First-wins meant a
    one-line abstract could squat on the address and the page fetched afterwards —
    the only version that answers anything — was dropped as a duplicate, leaving the
    ledger to report that it had nothing but abstracts for a page already read.
    """
    merged = list(existing or [])
    position_of: dict[str, int] = {}
    for position, source in enumerate(merged):
        key = _normalize_source_url(source.get("url", ""))
        if key:
            position_of.setdefault(key, position)

    for source in new_sources:
        key = _normalize_source_url(source.get("url", ""))
        if not key:
            continue
        position = position_of.get(key)
        if position is None:
            position_of[key] = len(merged)
            merged.append(source)
        elif _source_worth(source) > _source_worth(merged[position]):
            merged[position] = source
    return merged


_SOURCE_BLOCK_OVERHEAD_CHARS = 200  # the [n]/Title/URL/Published/Source type header
MIN_SOURCE_EXCERPT_CHARS = 2000  # below this an excerpt is too thin to judge a claim on


def _source_excerpt_budget(count: int) -> int:
    """Per-source excerpt length that lets ``count`` sources share the total budget.

    A fixed per-source cap meant the total budget ran out partway down the list and
    the rest of the fetched pages were dropped before the model ever saw them — 25
    pages read, 9 shown. Dividing the budget keeps every fetched page visible; the
    excerpt shrinks instead of the source list.
    """
    if count <= 0:
        return SOURCE_CONTENT_MAX_CHARS
    share = TOTAL_SOURCES_MAX_CHARS // count - _SOURCE_BLOCK_OVERHEAD_CHARS
    return max(MIN_SOURCE_EXCERPT_CHARS, min(SOURCE_CONTENT_MAX_CHARS, share))


# Kinds whose content is a body the agent actually fetched, as opposed to a record an
# index returned about it. Stated positively per source, so a list mixing the two cannot
# be read as if it held only index entries.
_FETCHED_KINDS = {
    "page",
    "table",
    "file",
    "json",
    "recipe_rows",
    "browser_table",
    "archive",
}


def _format_sources_for_llm(
    sources: list[dict], query: str = ""
) -> tuple[str, set[int]]:
    """Render sources for a prompt, keeping the part of each that bears on ``query``.

    Without a query — and before PASSAGE_RELEVANCE_EXCERPTS — an over-long page was cut
    to its first ``budget`` characters, which on a long page is the masthead and the
    navigation. The sentence the question turns on is past the cut, and the model is then
    asked why it could not find it.
    """
    parts = []
    valid_ids = set()
    total_chars = 0
    budget = _source_excerpt_budget(len(sources or []))
    by_relevance = bool(query) and PASSAGE_RELEVANCE_EXCERPTS

    for src_id, source in enumerate(sources or [], 1):
        raw_content = source.get("content", "")
        content = (
            _clip_text(relevant_excerpt(raw_content, query, budget), budget)
            if by_relevance
            else _clip_text(raw_content, budget)
        )
        if not content:
            continue
        source_type = (
            (source.get("source_quality") or {}).get("source_type")
            or source.get("kind")
            or "unknown"
        )
        if source.get("kind") in _FETCHED_KINDS:
            source_type = f"{source_type} — full text fetched"
        block = (
            f"[{src_id}]\n"
            f"Title: {source.get('title') or 'Untitled'}\n"
            f"URL: {source.get('url') or ''}\n"
            f"Published: {source.get('published') or 'unknown'}\n"
            f"Source type: {source_type}\n"
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
    return sum(
        1
        for source in sources or []
        if isinstance(source, dict) and source.get("kind") in structured_kinds
    )


def _audit_evidence_state(state: dict[str, Any]) -> dict:
    ledger_result = state.get("evidence_ledger_result")
    if isinstance(ledger_result, dict):
        admitted = state.get("admissible_sources", []) or state.get("sources", []) or []
        sources = [source for source in admitted if isinstance(source, dict)]
        candidate_count = int(
            ledger_result.get("candidate_count")
            or len(state.get("candidate_sources", []) or [])
        )
    else:
        sources = [
            source
            for source in state.get("sources", []) or []
            if isinstance(source, dict)
        ]
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
        return sum(
            len(table.get("rows", []) or [])
            for table in payload["tables"]
            if isinstance(table, dict)
        )
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
