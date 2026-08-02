"""Parsing of planner step strings into concrete tool calls."""

import ast
import re

from .llm import _json_loads_best_effort


def _normalize_tool_arguments(name: str, args: dict) -> dict:
    args = dict(args or {})
    if name == "check_date_completeness":
        if args.get("actual_items") is None:
            args["actual_items"] = []
        if args.get("holidays") is None:
            args["holidays"] = []
        args.setdefault("granularity", "day")
        args.setdefault("calendar", "calendar")
    return args


def _tool_call_from_python_like_step(step: str) -> dict | None:
    try:
        parsed = ast.parse(step.strip(), mode="eval")
    except SyntaxError:
        return None
    call = parsed.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    name = call.func.id
    if call.args:
        return None

    allowed = {
        "check_date_completeness",
        "browser_set_date_range",
        "browser_extract_tables_for_date_range",
        "validate_unit_rows",
        "web_fetch_json",
    }
    if name not in allowed:
        return None

    args = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        try:
            args[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return None
    return {"function": {"name": name, "arguments": _normalize_tool_arguments(name, args)}}


def _parse_step_kwargs(raw_arg: str) -> dict | None:
    """Parse a 'key=value, key=value' argument string into a dict.

    Returns None when the argument is not in that form (e.g. a bare URL or query),
    so callers can fall back to treating raw_arg as a single positional value.
    A bare URL never matches because it starts with 'scheme:' not 'key='.
    """
    if not re.match(r"^\s*\w+\s*=", raw_arg):
        return None
    kwargs: dict = {}
    for part in re.split(r",\s*(?=\w+\s*=)", raw_arg):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        low = value.lower()
        if low in ("true", "false"):
            kwargs[key] = (low == "true")
        elif value.isdigit():
            kwargs[key] = int(value)
        else:
            kwargs[key] = value
    return kwargs or None


def _tool_call_from_step(step: str) -> dict | None:
    python_like = _tool_call_from_python_like_step(step)
    if python_like:
        return python_like

    if ":" not in step:
        return None
    name, raw_arg = step.split(":", 1)
    name = name.strip()
    raw_arg = raw_arg.strip()
    if not raw_arg:
        return None

    # Model may emit "tool: url=..., lang=en, use_cache=false" instead of "tool: <url>".
    # Parse those kwargs so the URL/query isn't swallowed whole as one malformed value.
    kwargs = _parse_step_kwargs(raw_arg)
    if kwargs:
        url_tools = {"web_read", "web_extract_tables", "web_detect_downloads",
                     "web_parse_file", "web_fetch_json", "classify_source", "source_cache_get"}
        if name in url_tools and "url" in kwargs:
            return {"function": {"name": name, "arguments": kwargs}}
        if name in {"web_search", "web_deep_search"} and "query" in kwargs:
            return {"function": {"name": name, "arguments": kwargs}}

    if name == "web_search":
        return {"function": {"name": "web_search", "arguments": {"query": raw_arg, "num": 10}}}
    if name == "web_deep_search":
        return {"function": {"name": "web_deep_search", "arguments": {"query": raw_arg}}}
    if name == "web_read":
        return {"function": {"name": "web_read", "arguments": {"url": raw_arg}}}
    if name == "web_extract_tables":
        return {"function": {"name": "web_extract_tables", "arguments": {"url": raw_arg}}}
    if name == "web_detect_downloads":
        return {"function": {"name": "web_detect_downloads", "arguments": {"url": raw_arg}}}
    if name == "web_parse_file":
        return {"function": {"name": "web_parse_file", "arguments": {"url": raw_arg}}}
    if name == "web_fetch_json":
        return {"function": {"name": "web_fetch_json", "arguments": {"url": raw_arg}}}
    if name == "classify_source":
        return {"function": {"name": "classify_source", "arguments": {"url": raw_arg}}}
    if name == "generate_search_queries":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {"function": {"name": "generate_search_queries", "arguments": parsed}}
        return {"function": {"name": "generate_search_queries", "arguments": {"task": raw_arg}}}
    if name == "resolve_units":
        return {"function": {"name": "resolve_units", "arguments": {"text": raw_arg}}}
    if name == "validate_unit_rows":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "validate_unit_rows", "arguments": parsed}}
    if name == "source_cache_get":
        return {"function": {"name": "source_cache_get", "arguments": {"url": raw_arg}}}
    if name == "check_date_completeness":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "check_date_completeness", "arguments": _normalize_tool_arguments(name, parsed)}}
    if name == "evidence_entailment":
        if "||" in raw_arg:
            claim, excerpt = raw_arg.split("||", 1)
            return {
                "function": {
                    "name": "evidence_entailment",
                    "arguments": {"claim": claim.strip(), "source_excerpt": excerpt.strip()},
                }
            }
    if name == "tool_spec_propose":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {"function": {"name": "tool_spec_propose", "arguments": parsed}}
        return {"function": {"name": "tool_spec_propose", "arguments": {"task": raw_arg}}}
    if name == "tool_code_generate":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "tool_code_generate", "arguments": parsed}}
    if name == "tool_code_validate":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {"function": {"name": "tool_code_validate", "arguments": parsed}}
        return {"function": {"name": "tool_code_validate", "arguments": {"code": raw_arg}}}
    if name == "tool_code_run_sandboxed":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "tool_code_run_sandboxed", "arguments": parsed}}
    if name == "tool_promote":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "tool_promote", "arguments": parsed}}
    if name == "web_navigate":
        return {"function": {"name": "web_navigate", "arguments": {"url": raw_arg}}}
    if name == "browser_extract_tables":
        return {"function": {"name": "browser_extract_tables", "arguments": {}}}
    if name == "browser_set_date_range":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "browser_set_date_range", "arguments": parsed}}
    if name == "browser_extract_tables_for_date_range":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "browser_extract_tables_for_date_range", "arguments": parsed}}
    return None
