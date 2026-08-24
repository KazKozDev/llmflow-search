"""Parsing of planner step strings into concrete tool calls."""

import ast
import re

from jsonschema import Draft202012Validator

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


def _live_tool_schema(name: str, tools: list[dict]) -> dict | None:
    for tool in tools or []:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        if function.get("name") == name:
            schema = function.get("parameters")
            return schema if isinstance(schema, dict) else {"type": "object"}
    return None


_NAMED_ARG_LEAD = re.compile(r"\s*(\w+)\s*[:=]")

# Heads a model occasionally wraps a step in — "tool: web_read: <url>" means the same
# as "web_read: <url>"; the leading word is scaffolding, not a tool name. Nothing a
# planner could legitimately emit starts with these, so stripping is safe whenever the
# rest is itself a "name: arg" step.
_SPURIOUS_STEP_HEADS = {"tool", "function", "call", "mcp"}


def _strip_spurious_head(step: str) -> str:
    """Drop a leading 'tool:'-style wrapper from a step string if one is present."""
    head, sep, tail = step.partition(":")
    if sep and head.strip().lower() in _SPURIOUS_STEP_HEADS:
        inner, _, inner_arg = tail.partition(":")
        if inner.strip() and inner_arg.strip():
            return tail
    return step


def _coerce_to_schema(value: str, spec: dict):
    """Coerce a parsed kwarg string to the type its schema declares."""
    types = spec.get("type") if isinstance(spec, dict) else None
    types = types if isinstance(types, list) else [types]
    text = value.strip()
    if "boolean" in types and text.lower() in ("true", "false"):
        return text.lower() == "true"
    if "integer" in types:
        try:
            return int(text)
        except ValueError:
            return text
    if "number" in types:
        try:
            return float(text)
        except ValueError:
            return text
    return text


def _named_arg_lead(raw_arg: str, properties: dict) -> bool:
    """True when the argument opens with ``<real parameter>:`` or ``<real parameter>=``.

    Requiring the leading name to be an actual parameter is what keeps a bare URL
    from being read as named arguments — ``https://example.com`` opens with
    ``https:``, but no tool has a parameter called ``https``.
    """
    match = _NAMED_ARG_LEAD.match(raw_arg)
    return bool(match and match.group(1) in (properties or {}))


def _kwargs_from_schema(raw_arg: str, properties: dict) -> dict | None:
    """Parse named arguments against a tool's schema, ``key: value`` or ``key=value``.

    Models write this form constantly even when told not to, and they switch freely
    between the two separators. Splitting on the *parameter names the schema declares*
    rather than on a token pattern means an unquoted multi-word value survives intact:
    ``query: python 3.14 release notes`` is one value, not a value plus leftovers. The
    alternative was the single-required-parameter fallback swallowing the whole string,
    so the tool ran on the literal text ``query: '...' lang:en num:10``.
    """
    if not properties:
        return None
    if not _named_arg_lead(raw_arg, properties):
        return None

    # Longest names first so a parameter that prefixes another still matches whole.
    names = "|".join(
        re.escape(name) for name in sorted(properties, key=len, reverse=True)
    )
    boundary = re.compile(rf"(?:^|[\s,])({names})\s*[:=]\s*")
    matches = list(boundary.finditer(raw_arg))
    if not matches or matches[0].start() != 0:
        return None

    kwargs: dict = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_arg)
        value = raw_arg[match.end() : end].strip().strip(",").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value:
            return None
        kwargs[match.group(1)] = _coerce_to_schema(
            value, properties.get(match.group(1)) or {}
        )
    return kwargs or None


def _tool_call_from_schema_step(step: str, tools: list[dict]) -> dict | None:
    """Resolve any live MCP tool from ``name: argument`` using its JSON Schema.

    A JSON object is used verbatim, ``key=value`` pairs are resolved against the
    schema, and a bare value is accepted only when the schema has exactly one
    required parameter (or exactly one property), which keeps simple URL and query
    steps concise without hard-coding tool names.
    """
    if ":" not in step:
        return None
    step = _strip_spurious_head(step)
    name, raw_arg = step.split(":", 1)
    name = name.strip()
    raw_arg = raw_arg.strip()
    schema = _live_tool_schema(name, tools)
    if schema is None:
        return None

    raw_properties = schema.get("properties")
    properties: dict = raw_properties if isinstance(raw_properties, dict) else {}

    sentinel = object()
    parsed = _json_loads_best_effort(raw_arg, sentinel) if raw_arg else {}
    if isinstance(parsed, dict):
        arguments = parsed
    elif (kwargs := _kwargs_from_schema(raw_arg, properties)) is not None:
        arguments = kwargs
    elif _named_arg_lead(raw_arg, properties):
        # Named arguments we could not resolve. Passing the whole string through as
        # one value is worse than failing: the tool would run on literal text like
        # "query: '...' lang:en num:10" and return nothing usable.
        return None
    else:
        raw_required = schema.get("required")
        required = [
            item
            for item in (raw_required if isinstance(raw_required, list) else [])
            if item in properties
        ]
        candidates = (
            required
            if len(required) == 1
            else list(properties)
            if len(properties) == 1
            else []
        )
        if len(candidates) != 1 or not raw_arg:
            return None
        value = raw_arg if parsed is sentinel else parsed
        arguments = {candidates[0]: value}

    arguments = _normalize_tool_arguments(name, arguments)
    try:
        Draft202012Validator(schema).validate(arguments)
    except Exception:
        return None
    return {"function": {"name": name, "arguments": arguments}}


def _step_identity(step: str, tools: list[dict] | None = None) -> str:
    """Canonical identity of a step, so one call written two ways dedups as one.

    Keyed on the resolved tool plus its *required* arguments: ``web_search: X`` and
    ``web_search: query: 'X' lang:en num:10`` are the same attempt, and a retry that
    only changes ``num`` is not progress. Unresolvable steps fall back to their
    whitespace-normalized text.
    """
    call = _tool_call_from_schema_step(step, tools or [])
    if not call:
        name, _, raw_arg = step.partition(":")
        return f"{name.strip().lower()}:{' '.join(raw_arg.split()).lower()}"

    function = call.get("function", {})
    name = str(function.get("name", "")).lower()
    arguments = function.get("arguments") or {}
    schema = _live_tool_schema(name, tools or []) or {}
    raw_required = schema.get("required")
    required = [
        item
        for item in (raw_required if isinstance(raw_required, list) else [])
        if item in arguments
    ]
    keys = required or sorted(arguments)
    signature = " ".join(
        f"{key}={' '.join(str(arguments[key]).split())}" for key in sorted(keys)
    )
    return f"{name}:{signature}".lower()


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
    return {
        "function": {"name": name, "arguments": _normalize_tool_arguments(name, args)}
    }


def _parse_step_kwargs(raw_arg: str) -> dict | None:
    """Parse a 'key=value' argument string, comma- or space-separated, into a dict.

    Returns None when the argument is not in that form (e.g. a bare URL or query),
    so callers can fall back to treating raw_arg as a single positional value.
    A bare URL never matches because it starts with 'scheme:' not 'key='.
    """
    if not re.match(r"^\s*\w+\s*=", raw_arg):
        return None
    kwargs: dict = {}
    for match in re.finditer(r"""(\w+)\s*=\s*('[^']*'|"[^"]*"|[^\s,]+)""", raw_arg):
        key = match.group(1)
        value = match.group(2)
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        low = value.lower()
        if low in ("true", "false"):
            kwargs[key] = low == "true"
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
    step = _strip_spurious_head(step)
    name, raw_arg = step.split(":", 1)
    name = name.strip()
    raw_arg = raw_arg.strip()
    if not raw_arg:
        return None

    # Model may emit "tool: url=..., lang=en, use_cache=false" instead of "tool: <url>".
    # Parse those kwargs so the URL/query isn't swallowed whole as one malformed value.
    kwargs = _parse_step_kwargs(raw_arg)
    if kwargs:
        url_tools = {
            "web_read",
            "web_extract_tables",
            "web_detect_downloads",
            "web_parse_file",
            "web_fetch_json",
            "classify_source",
            "source_cache_get",
        }
        if name in url_tools and "url" in kwargs:
            return {"function": {"name": name, "arguments": kwargs}}
        if name in {"web_search", "web_deep_search"} and "query" in kwargs:
            return {"function": {"name": name, "arguments": kwargs}}

    if name == "web_search":
        return {
            "function": {
                "name": "web_search",
                "arguments": {"query": raw_arg, "num": 10},
            }
        }
    if name == "web_deep_search":
        return {
            "function": {"name": "web_deep_search", "arguments": {"query": raw_arg}}
        }
    if name == "web_read":
        return {"function": {"name": "web_read", "arguments": {"url": raw_arg}}}
    if name == "web_extract_tables":
        return {
            "function": {"name": "web_extract_tables", "arguments": {"url": raw_arg}}
        }
    if name == "web_detect_downloads":
        return {
            "function": {"name": "web_detect_downloads", "arguments": {"url": raw_arg}}
        }
    if name == "web_parse_file":
        return {"function": {"name": "web_parse_file", "arguments": {"url": raw_arg}}}
    if name == "web_fetch_json":
        return {"function": {"name": "web_fetch_json", "arguments": {"url": raw_arg}}}
    if name == "classify_source":
        return {"function": {"name": "classify_source", "arguments": {"url": raw_arg}}}
    if name == "generate_search_queries":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {
                "function": {"name": "generate_search_queries", "arguments": parsed}
            }
        return {
            "function": {
                "name": "generate_search_queries",
                "arguments": {"task": raw_arg},
            }
        }
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
            return {
                "function": {
                    "name": "check_date_completeness",
                    "arguments": _normalize_tool_arguments(name, parsed),
                }
            }
    if name == "evidence_entailment":
        if "||" in raw_arg:
            claim, excerpt = raw_arg.split("||", 1)
            return {
                "function": {
                    "name": "evidence_entailment",
                    "arguments": {
                        "claim": claim.strip(),
                        "source_excerpt": excerpt.strip(),
                    },
                }
            }
    if name == "tool_spec_propose":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {"function": {"name": "tool_spec_propose", "arguments": parsed}}
        return {
            "function": {"name": "tool_spec_propose", "arguments": {"task": raw_arg}}
        }
    if name == "tool_code_generate":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {"function": {"name": "tool_code_generate", "arguments": parsed}}
    if name == "tool_code_validate":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict) and parsed:
            return {"function": {"name": "tool_code_validate", "arguments": parsed}}
        return {
            "function": {"name": "tool_code_validate", "arguments": {"code": raw_arg}}
        }
    if name == "tool_code_run_sandboxed":
        parsed = _json_loads_best_effort(raw_arg, {})
        if isinstance(parsed, dict):
            return {
                "function": {"name": "tool_code_run_sandboxed", "arguments": parsed}
            }
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
            return {
                "function": {
                    "name": "browser_extract_tables_for_date_range",
                    "arguments": parsed,
                }
            }
    return None
