"""Per-step observation normalization and model-assisted diagnosis."""

import json
import re

from . import llm
from .llm import _json_loads_best_effort
from .memory import _slug_key
from .prompts import OBSERVATION_SYSTEM_PROMPT
from .search_memory import _append_unique, _merge_search_memory
from .sources import _normalize_source_url, _payload_row_count

ALLOWED_OBSERVATION_ACTION_TAGS = {
    "search_better_sources",
    "search_structured_sources",
    "search_machine_readable",
    "browser_fallback",
    "recipe_candidate",
    "refine_query",
    "stop_and_answer",
}


def _compact_payload_for_observation(payload: dict, max_chars: int = 4000) -> dict:
    compact = {}
    for key in (
        "url",
        "title",
        "pub_date",
        "published",
        "source_type",
        "content_type",
        "status_code",
        "error",
        "count",
        "table_count",
        "file_type",
    ):
        if key in payload:
            compact[key] = payload[key]
    if isinstance(payload.get("text"), str):
        compact["text"] = payload["text"][:max_chars]
    if isinstance(payload.get("tables"), list):
        compact["tables"] = payload["tables"][:2]
    if isinstance(payload.get("sources"), list):
        compact["sources"] = []
        for source in payload["sources"][:10]:
            if not isinstance(source, dict):
                continue
            compact["sources"].append(
                {
                    key: source[key]
                    for key in ("url", "title", "snippet", "pub_date", "published", "source_type")
                    if key in source
                }
            )
    if isinstance(payload.get("downloads"), list):
        compact["downloads"] = payload["downloads"][:10]
    if "json" in payload:
        compact["json_preview"] = json.dumps(payload.get("json"), ensure_ascii=False)[:max_chars]
    if isinstance(payload.get("result"), dict):
        compact["result"] = payload["result"]
    return compact


def _normalize_observation(
    raw: dict | None,
    tool_name: str,
    args: dict,
    payload: dict,
    current_step: str,
) -> dict:
    payload = payload if isinstance(payload, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    source_url = _normalize_source_url(str(args.get("url") or payload.get("url") or ""))
    row_count = _payload_row_count(payload)
    is_structured = tool_name in {"web_extract_tables", "web_parse_file", "web_fetch_json", "tool_code_run_sandboxed", "browser_extract_tables"}
    has_rows = row_count > 0 or bool(payload.get("json"))
    source_type = str(raw.get("source_quality") or raw.get("source_type") or "unknown").lower().strip()
    if source_type not in {"primary", "secondary", "aggregator", "blog", "forum", "interactive", "blocked", "unknown"}:
        source_type = "unknown"
    gaps = [str(gap).strip()[:120] for gap in raw.get("gaps", []) if str(gap).strip()]
    next_action_tags = [
        str(tag).strip()
        for tag in raw.get("next_action_tags", [])
        if str(tag).strip() in ALLOWED_OBSERVATION_ACTION_TAGS
    ]
    suggested_queries = []
    raw_queries = raw.get("query_candidates", raw.get("suggested_queries", []))
    if not isinstance(raw_queries, list):
        raw_queries = []
    for query in raw_queries:
        query = re.sub(r"\s+", " ", str(query)).strip()
        if query and query not in suggested_queries:
            suggested_queries.append(query[:300])
    urls = []
    raw_urls = raw.get("urls", [])
    if not isinstance(raw_urls, list):
        raw_urls = []
    for url in raw_urls:
        url = _normalize_source_url(str(url))
        if url and url not in urls:
            urls.append(url[:500])
    if source_url and source_url not in urls:
        urls.insert(0, source_url)
    publication_dates = [
        str(date).strip()[:40]
        for date in raw.get("publication_dates", [])
        if str(date).strip()
    ] if isinstance(raw.get("publication_dates"), list) else []
    if raw.get("publication_date"):
        publication_dates.insert(0, str(raw["publication_date"]).strip()[:40])
    event_dates = [
        str(date).strip()[:40]
        for date in raw.get("event_dates", [])
        if str(date).strip()
    ] if isinstance(raw.get("event_dates"), list) else []
    errors = [
        str(error).strip()[:200]
        for error in raw.get("errors", [])
        if str(error).strip()
    ] if isinstance(raw.get("errors"), list) else []

    return {
        "tool": tool_name,
        "step": current_step,
        "url": source_url,
        "source_type": source_type,
        "useful": bool(raw.get("useful")),
        "dated": bool(raw.get("dated")),
        "structured": bool(raw.get("structured", is_structured)),
        "has_rows": bool(raw.get("has_rows", has_rows)),
        "row_count": row_count,
        "gaps": list(dict.fromkeys(gaps)),
        "next_action_tags": list(dict.fromkeys(next_action_tags)),
        "suggested_queries": suggested_queries[:4],
        "query_candidates": suggested_queries[:4],
        "summary": str(raw.get("summary") or "").strip()[:700],
        "source_title": str(raw.get("source_title") or raw.get("title") or payload.get("title") or "").strip()[:200],
        "publication_dates": list(dict.fromkeys(publication_dates))[:5],
        "event_dates": list(dict.fromkeys(event_dates))[:5],
        "urls": urls[:10],
        "errors": list(dict.fromkeys(errors))[:5],
        "failure_diagnosis": str(raw.get("failure_diagnosis") or "").strip()[:500],
        "reason": str(raw.get("reason") or "Observation diagnosis unavailable.")[:500],
    }


def _fallback_observation(tool_name: str, args: dict, payload: dict, current_step: str) -> dict:
    # Used only when the LLM observation diagnosis is unavailable. Records objective
    # facts about the tool result (did content come back) and leaves all strategy
    # decisions — gaps and next actions — to the LLM. No deterministic strategy inference.
    payload = payload if isinstance(payload, dict) else {}
    row_count = _payload_row_count(payload)
    has_rows = row_count > 0 or bool(payload.get("json"))
    if payload.get("error"):
        useful = False
    elif tool_name in {"web_search", "web_deep_search"}:
        useful = bool(payload.get("count") or payload.get("source_count"))
    elif tool_name == "web_read":
        useful = bool(payload.get("text"))
    else:
        useful = has_rows
    raw = {
        "useful": useful,
        "structured": tool_name in {"web_extract_tables", "web_parse_file", "web_fetch_json", "tool_code_run_sandboxed", "browser_extract_tables"},
        "has_rows": has_rows,
        "dated": bool(payload.get("pub_date") or payload.get("published")),
        "source_quality": "unknown",
        "gaps": [],
        "next_action_tags": [],
        "suggested_queries": [],
        "summary": "",
        "source_title": payload.get("title", ""),
        "publication_dates": [payload.get("pub_date") or payload.get("published")] if (payload.get("pub_date") or payload.get("published")) else [],
        "event_dates": [],
        "urls": [],
        "errors": [str(payload.get("error"))] if payload.get("error") else [],
        "failure_diagnosis": "",
        "reason": "Objective fallback (LLM diagnosis unavailable).",
    }
    return _normalize_observation(raw, tool_name, args, payload, current_step)


def _diagnose_observation_with_model(
    model: str,
    tool_name: str,
    args: dict,
    payload: dict,
    *,
    question: str,
    requirements: dict,
    current_step: str,
) -> dict:
    fallback = _fallback_observation(tool_name, args, payload, current_step)
    prompt = f"""QUESTION:
{question}

TASK_REQUIREMENTS:
{json.dumps(requirements, ensure_ascii=False, indent=2)}

CURRENT_STEP:
{current_step}

TOOL_NAME:
{tool_name}

TOOL_ARGUMENTS:
{json.dumps(args, ensure_ascii=False, indent=2)}

TOOL_RESULT_PREVIEW:
{json.dumps(_compact_payload_for_observation(payload), ensure_ascii=False, indent=2)}

Diagnose this observation and choose the next search action tags."""
    try:
        response = llm._ollama_chat(
            model,
            [{"role": "user", "content": prompt}],
            tools=None, json_mode=True,
            system=OBSERVATION_SYSTEM_PROMPT,
            temperature=0,
        )
        raw = _json_loads_best_effort(response.get("content", ""), {})
        if isinstance(raw, dict) and raw:
            observation = _normalize_observation(raw, tool_name, args, payload, current_step)
            return observation
    except Exception:
        return fallback
    return fallback


def _enrich_sources_with_observation(sources: list[dict], observation: dict | None) -> list[dict]:
    if not observation:
        return sources
    enriched = []
    source_type = str(observation.get("source_type") or "").strip()
    publication_dates = observation.get("publication_dates") if isinstance(observation.get("publication_dates"), list) else []
    for source in sources:
        if not isinstance(source, dict):
            enriched.append(source)
            continue
        updated = dict(source)
        if observation.get("source_title") and (not updated.get("title") or updated.get("title") == updated.get("url")):
            updated["title"] = observation["source_title"]
        if publication_dates and not updated.get("published"):
            updated["published"] = publication_dates[0]
        if source_type and source_type != "unknown":
            quality = updated.get("source_quality")
            if not isinstance(quality, dict):
                quality = {}
            updated["source_quality"] = quality | {"source_type": source_type}
        if observation.get("summary"):
            updated["tool_observation_summary"] = observation["summary"]
        if observation.get("event_dates"):
            updated["event_dates"] = observation["event_dates"]
        if observation.get("errors"):
            updated["tool_errors"] = observation["errors"]
        enriched.append(updated)
    return enriched


def _record_observation(memory: dict | None, observation: dict, requirements: dict) -> dict:
    memory = _merge_search_memory(memory)
    compact = {
        "tool": observation.get("tool"),
        "url": observation.get("url"),
        "useful": observation.get("useful"),
        "dated": observation.get("dated"),
        "structured": observation.get("structured"),
        "has_rows": observation.get("has_rows"),
        "gaps": observation.get("gaps", []),
        "next_action_tags": observation.get("next_action_tags", []),
        "summary": observation.get("summary", ""),
        "source_title": observation.get("source_title", ""),
        "publication_dates": observation.get("publication_dates", []),
        "event_dates": observation.get("event_dates", []),
        "urls": observation.get("urls", []),
        "errors": observation.get("errors", []),
        "failure_diagnosis": observation.get("failure_diagnosis", ""),
        "query_candidates": observation.get("query_candidates", observation.get("suggested_queries", [])),
        "reason": observation.get("reason", ""),
    }
    memory["observations"].append(compact)
    memory["observations"] = memory["observations"][-50:]
    for gap in observation.get("gaps", []):
        _append_unique(memory["open_gaps"], gap)
    if observation.get("useful"):
        for gap in observation.get("gaps", []):
            if gap in memory["open_gaps"]:
                memory["open_gaps"].remove(gap)
    if observation.get("url") and not observation.get("useful"):
        _append_unique(memory["avoid_urls"], observation["url"])
        m = re.search(r"https?://(?:www\.)?([^/]+)", observation["url"])
        if m:
            _append_unique(memory.setdefault("bad_domains", []), m.group(1).lower())
    for tag in observation.get("next_action_tags", []):
        _append_unique(memory["next_actions"], tag)
    for query in observation.get("query_candidates", observation.get("suggested_queries", [])):
        _append_unique(memory.setdefault("query_candidates", []), query)
    failure_diagnosis = str(observation.get("failure_diagnosis") or "").strip()
    if failure_diagnosis:
        _append_unique(memory.setdefault("observation_failure_diagnoses", []), failure_diagnosis)
    if observation.get("useful") and observation.get("url"):
        criteria = requirements.get("completion_criteria") if isinstance(requirements, dict) else []
        key = _slug_key(" | ".join(criteria[:2]) if criteria else requirements.get("target", "evidence"))
        evidence = memory["evidence_map"].setdefault(key, [])
        entry = {
            "url": observation["url"],
            "tool": observation.get("tool"),
            "dated": observation.get("dated"),
            "structured": observation.get("structured"),
            "row_count": observation.get("row_count", 0),
        }
        if entry not in evidence:
            evidence.append(entry)
    return memory
