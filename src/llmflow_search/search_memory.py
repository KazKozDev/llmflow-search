"""Live search-attempt memory and strategy-from-memory planning."""

import json

from . import memory as _memmod
from .llm import _json_loads_best_effort
from .sources import _normalize_source_url


def _default_search_memory() -> dict:
    return {
        "attempted_queries": [],
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
        "empty_structured_attempts": [],
        "observations": [],
        "open_gaps": [],
        "next_actions": [],
        "query_candidates": [],
        "observation_failure_diagnoses": [],
        "avoid_urls": [],
        "bad_domains": [],
        "evidence_map": {},
    }


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


def _update_search_memory(memory: dict | None, tool_name: str, args: dict, tool_result: str) -> dict:
    memory = _merge_search_memory(memory)
    payload = _json_loads_best_effort(tool_result, {})
    if not isinstance(payload, dict):
        payload = {}

    if tool_name in ("web_search", "web_deep_search"):
        query = str(args.get("query", "")).strip()
        _append_unique(memory["attempted_queries"], query)
        memory["search_rounds"].append(
            {
                "tool": tool_name,
                "query": query,
                "result_count": payload.get("count") or payload.get("source_count") or 0,
                "source_count": payload.get("source_count") or 0,
            }
        )
        for source in payload.get("sources", []):
            if isinstance(source, dict):
                _append_unique(memory["read_urls"], _normalize_source_url(source.get("url", "")))

    if tool_name == "web_read":
        url = _normalize_source_url(args.get("url", ""))
        if payload.get("error"):
            _append_unique(memory["failed_urls"], url)
        elif payload.get("text"):
            _append_unique(memory["read_urls"], url)
        else:
            _append_unique(memory["failed_urls"], url)

    if tool_name in ("web_extract_tables", "web_parse_file", "web_fetch_json", "tool_code_run_sandboxed", "browser_extract_tables"):
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
            table_count = payload.get("table_count") or len(payload.get("tables", []) or [])
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
            if tool_name in {"web_extract_tables", "browser_extract_tables"} and table_count == 0:
                _append_unique(memory["empty_structured_attempts"], url)
            if tool_name == "web_parse_file" and not payload.get("tables") and "json" not in payload:
                _append_unique(memory["empty_structured_attempts"], url)

    if tool_name == "web_detect_downloads":
        url = _normalize_source_url(args.get("url", "") or payload.get("url", ""))
        if payload.get("error"):
            _append_unique(memory["failed_urls"], url)
        elif (payload.get("count") or len(payload.get("downloads", []) or [])) == 0:
            _append_unique(memory["empty_structured_attempts"], url)

    return memory


def _strategy_plan_from_memory(question: str, memory: dict | None) -> list[str]:
    memory = _merge_search_memory(memory)
    queries = [query for query in memory.get("next_queries", []) if query]
    # Only search steps. After the search batch, the post-batch LLM decides which
    # result URLs to read (NEXT → web_read: <url>). No placeholder scaffolding.
    return [f"web_search: {query}" for query in queries[:4]]


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
            "query_candidates_from_observations": memory.get("query_candidates", [])[-10:],
            "observation_failure_diagnoses": memory.get("observation_failure_diagnoses", [])[-10:],
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
            "barren_domains": list({d for sk in store.get_skills(limit=20) for d in sk.get("barren_domains", [])})[:15],
        },
        ensure_ascii=False,
        indent=2,
    )
