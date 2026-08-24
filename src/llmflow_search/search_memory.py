"""Live search-attempt memory and strategy-from-memory planning."""

import json

from . import memory as _memmod
from .config import SNIPPET_MEMORY_MAX_CHARS
from .llm import _json_loads_best_effort
from .sources import _normalize_source_url


def _default_search_memory() -> dict:
    return {
        "attempted_queries": [],
        # URLs a search actually returned. Kept apart from read_urls so a step
        # can be checked against what exists before it is executed.
        "discovered_urls": [],
        "discovered_titles": {},
        # The snippet a search returned for a URL, and the best rank it ever held.
        # Without them the model is asked which page to read from a bare address list.
        "discovered_snippets": {},
        "discovered_ranks": {},
        "read_urls": [],
        "failed_urls": [],
        "search_rounds": [],
        "strategy_notes": [],
        "reflections": [],
        "next_queries": [],
        "current_strategy": "",
        "strategy_candidates": [],
        "search_hypotheses": [],
        "failure_diagnoses": [],
        "mutation_history": [],
        "exhausted_directions": [],
        "search_exhausted": False,
        # Rate-limited calls spent on the current question, checked against the budget.
        "search_calls": 0,
        "empty_structured_attempts": [],
        "observations": [],
        "open_gaps": [],
        "next_actions": [],
        "query_candidates": [],
        "observation_failure_diagnoses": [],
        "avoid_urls": [],
        "bad_domains": [],
        # Address of the page the interactive browser session is currently on. It is
        # the only provenance web_extract/web_snapshot output can be attributed to.
        "browser_url": "",
        "evidence_map": {},
    }


# Every tool that returns a normalized {results: [{url, title, ...}]} discovery list.
# Their URLs are real addresses, so a later fetch step naming one is not an invention.
_DISCOVERY_SEARCH_TOOLS = (
    "web_search",
    "web_deep_search",
    "web_search_recent",
    "papers_search",
    "encyclopedia_search",
    "github_search",
    "archive_search",
)


def _append_unique(items: list, value: str) -> list:
    value = (value or "").strip()
    if value and value not in items:
        items.append(value)
    return items


def _merge_search_memory(memory: dict | None) -> dict:
    merged = _default_search_memory()
    if isinstance(memory, dict):
        for key in merged:
            if isinstance(memory.get(key), list):
                merged[key] = list(memory[key])
            elif key in memory and not isinstance(merged.get(key), list):
                merged[key] = memory[key]
    return merged


def _update_search_memory(
    memory: dict | None, tool_name: str, args: dict, tool_result: str
) -> dict:
    memory = _merge_search_memory(memory)
    payload = _json_loads_best_effort(tool_result, {})
    if not isinstance(payload, dict):
        payload = {}

    if tool_name in _DISCOVERY_SEARCH_TOOLS:
        # archive_search is keyed by the URL being looked up, not by a query string.
        query = str(args.get("query", "") or args.get("url", "")).strip()
        _append_unique(memory["attempted_queries"], query)
        memory["search_rounds"].append(
            {
                "tool": tool_name,
                "query": query,
                "result_count": payload.get("count")
                or payload.get("source_count")
                or 0,
                "source_count": payload.get("source_count") or 0,
            }
        )
        # web_search returns "results"; web_deep_search returns "sources".
        for rank, source in enumerate(
            list(payload.get("results") or []) + list(payload.get("sources") or [])
        ):
            if not isinstance(source, dict):
                continue
            url = _normalize_source_url(source.get("url", ""))
            if not url:
                continue
            _append_unique(memory.setdefault("discovered_urls", []), url)
            title = str(source.get("title") or "").strip()
            if title:
                memory.setdefault("discovered_titles", {}).setdefault(url, title)
            snippet = " ".join(
                str(
                    source.get("snippet")
                    or source.get("description")
                    or source.get("text")
                    or ""
                ).split()
            )
            if snippet:
                memory.setdefault("discovered_snippets", {}).setdefault(
                    url, snippet[:SNIPPET_MEMORY_MAX_CHARS]
                )
            ranks = memory.setdefault("discovered_ranks", {})
            if rank < ranks.get(url, rank + 1):
                ranks[url] = rank

    if tool_name == "web_read":
        url = _normalize_source_url(args.get("url", ""))
        if payload.get("error"):
            _append_unique(memory["failed_urls"], url)
        elif payload.get("text"):
            _append_unique(memory["read_urls"], url)
        else:
            _append_unique(memory["failed_urls"], url)
        # Hyperlinks on a fetched page are real addresses too, so following one
        # is not an invention.
        for link in payload.get("links") or []:
            if isinstance(link, dict):
                _append_unique(
                    memory.setdefault("discovered_urls", []),
                    _normalize_source_url(link.get("url", "")),
                )

    if tool_name in (
        "web_extract_tables",
        "web_parse_file",
        "web_fetch_json",
        "tool_code_run_sandboxed",
        "browser_extract_tables",
    ):
        url = _normalize_source_url(args.get("url", "") or payload.get("url", ""))
        if tool_name == "tool_code_run_sandboxed":
            _r1 = payload.get("result")
            _result1: dict = _r1 if isinstance(_r1, dict) else {}
            _rr1 = _result1.get("rows")
            _rows1: list = _rr1 if isinstance(_rr1, list) else []
            for row in _rows1:
                if isinstance(row, dict) and row.get("source_url"):
                    url = _normalize_source_url(row["source_url"])
                    break
        if payload.get("error"):
            _append_unique(memory["failed_urls"], url)
        elif url:
            _append_unique(memory["read_urls"], url)
            table_count = payload.get("table_count") or len(
                payload.get("tables", []) or []
            )
            if tool_name == "tool_code_run_sandboxed":
                _r2 = payload.get("result")
                _result2: dict = _r2 if isinstance(_r2, dict) else {}
                _rr2 = _result2.get("rows")
                table_count = len(_rr2 if isinstance(_rr2, list) else [])
            memory["search_rounds"].append(
                {
                    "tool": tool_name,
                    "url": url,
                    "table_count": table_count,
                    "file_type": payload.get("file_type"),
                    "json": "json" in payload,
                }
            )
            if (
                tool_name in {"web_extract_tables", "browser_extract_tables"}
                and table_count == 0
            ):
                _append_unique(memory["empty_structured_attempts"], url)
            if (
                tool_name == "web_parse_file"
                and not payload.get("tables")
                and "json" not in payload
            ):
                _append_unique(memory["empty_structured_attempts"], url)

    if tool_name == "web_detect_downloads":
        url = _normalize_source_url(args.get("url", "") or payload.get("url", ""))
        downloads = payload.get("downloads") or []
        if payload.get("error"):
            _append_unique(memory["failed_urls"], url)
        elif (payload.get("count") or len(downloads)) == 0:
            _append_unique(memory["empty_structured_attempts"], url)
        # A detected file link is a real address found on a real page. Without this the
        # URL gate discards the web_parse_file step that this tool exists to enable.
        for item in downloads:
            if isinstance(item, dict):
                link = _normalize_source_url(item.get("url", ""))
                _append_unique(memory.setdefault("discovered_urls", []), link)
                text = str(item.get("text") or "").strip()
                if link and text:
                    memory.setdefault("discovered_titles", {}).setdefault(link, text)

    if tool_name == "web_crawl":
        for page in payload.get("pages") or []:
            if not isinstance(page, dict):
                continue
            page_url = _normalize_source_url(page.get("url", ""))
            if not page_url:
                continue
            if page.get("error"):
                _append_unique(memory["failed_urls"], page_url)
                continue
            _append_unique(memory.setdefault("discovered_urls", []), page_url)
            if page.get("text"):
                _append_unique(memory["read_urls"], page_url)
            title = str(page.get("title") or "").strip()
            if title:
                memory.setdefault("discovered_titles", {}).setdefault(page_url, title)

    if tool_name == "web_archive_fetch":
        original = _normalize_source_url(args.get("url", "") or payload.get("url", ""))
        snapshot = _normalize_source_url(payload.get("snapshot_url", ""))
        if payload.get("error") or payload.get("fetch_error"):
            _append_unique(memory["failed_urls"], original)
        if snapshot:
            _append_unique(memory.setdefault("discovered_urls", []), snapshot)
            if payload.get("text"):
                _append_unique(memory["read_urls"], snapshot)

    if tool_name in ("web_navigate", "web_snapshot"):
        page_url = _normalize_source_url(payload.get("url", "") or args.get("url", ""))
        if page_url:
            memory["browser_url"] = page_url
            _append_unique(memory.setdefault("discovered_urls", []), page_url)

    return memory


def _strategy_plan_from_memory(
    question: str, memory: dict | None, tool_names: set[str] | None = None
) -> list[str]:
    memory = _merge_search_memory(memory)
    queries = [query for query in memory.get("next_queries", []) if query]
    live = tool_names or set()
    steps: list[str] = []
    for query in queries[:4]:
        # The strategy node may change the access path, not only the wording — a query
        # already written as "papers_search: ..." is kept as that tool's step. A plain
        # query stays a web_search. After the batch the post-batch LLM decides what to
        # fetch from the results; no placeholder scaffolding here either way.
        head, _, tail = query.partition(":")
        if tail.strip() and head.strip() in live:
            steps.append(f"{head.strip()}: {tail.strip()}")
        else:
            steps.append(f"web_search: {query}")
    return steps


def _format_search_memory_for_prompt(memory: dict | None) -> str:
    memory = _merge_search_memory(memory)
    store = _memmod._get_research_store()
    return json.dumps(
        {
            "attempted_queries": memory["attempted_queries"][-20:],
            "read_urls": memory["read_urls"][-20:],
            "failed_urls": memory["failed_urls"][-20:],
            "unhelpful_urls_this_session": memory.get("avoid_urls", [])[-15:],
            "unhelpful_domains_this_session": memory.get("bad_domains", [])[-15:],
            "empty_structured_attempts": memory["empty_structured_attempts"][-10:],
            "observations": memory["observations"][-10:],
            "open_gaps": memory["open_gaps"][-10:],
            "next_actions": memory["next_actions"][-10:],
            "query_candidates_from_observations": memory.get("query_candidates", [])[
                -10:
            ],
            "observation_failure_diagnoses": memory.get(
                "observation_failure_diagnoses", []
            )[-10:],
            "evidence_map": memory["evidence_map"],
            "recent_search_rounds": memory["search_rounds"][-10:],
            "strategy_notes": memory["strategy_notes"][-5:],
            "search_hypotheses": memory["search_hypotheses"][-5:],
            "failure_diagnoses": memory["failure_diagnoses"][-5:],
            "mutation_history": memory["mutation_history"][-8:],
            "exhausted_directions": memory["exhausted_directions"][-8:],
            "search_exhausted": memory.get("search_exhausted", False),
            "proven_strategies": store.get_strategies(limit=5),
            "reusable_skills": store.get_skills(limit=5),
            "barren_domains": list(
                {
                    d
                    for sk in store.get_skills(limit=20)
                    for d in sk.get("barren_domains", [])
                }
            )[:15],
        },
        ensure_ascii=False,
        indent=2,
    )
