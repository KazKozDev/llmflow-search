"""Does the object a model returned actually match the schema it was given?

Ollama's ``format=<schema>`` argument is a request, not a guarantee. Only the
GGUF/llama.cpp decoder enforces it; the MLX runtime ignores it (see
``llm._schema_capable_model``) and so does every ``:cloud`` model, which is served by a
remote API that receives the messages and drops the schema. On those backends a caller
that passes a schema gets ordinary free-running generation and no error, so a
malformed evidence ledger looks exactly like an empty one — the run degrades quietly
into "no supported claims" and the trace records nothing about why.

This module makes that failure observable and fixable without a backend: a checker
strict enough to catch the shapes the pipeline's normalizers silently discard, and a
rendering of the schema that can be put in the prompt where an unenforcing backend will
at least read it.

Deliberately not a full JSON Schema implementation. It covers the constructs this
codebase's schemas actually use — object/array/string/integer/number/boolean types,
``required``, ``enum``, ``additionalProperties: false`` and integer bounds — and ignores
the rest rather than pretending to validate it.
"""

from typing import Any

_TYPE_CHECKS: dict[str, Any] = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    # JSON has one number type and bool is an int in Python; both distinctions matter
    # here, because a ledger row's requirement_index is an array position.
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
}

MAX_VIOLATIONS = 20


def schema_violations(value: Any, schema: dict, path: str = "$") -> list[str]:
    """Every way ``value`` departs from ``schema``, as human-readable paths.

    Returns an empty list for a conforming value. The list is capped: the caller uses it
    to repair a response or to record why one was rejected, and neither needs the
    hundredth violation.
    """
    violations: list[str] = []
    _collect(value, schema, path, violations)
    return violations[:MAX_VIOLATIONS]


def conforms(value: Any, schema: dict) -> bool:
    return not schema_violations(value, schema)


def _collect(value: Any, schema: dict, path: str, out: list[str]) -> None:
    if len(out) >= MAX_VIOLATIONS or not isinstance(schema, dict):
        return

    expected = schema.get("type")
    if isinstance(expected, str):
        check = _TYPE_CHECKS.get(expected)
        if check and not check(value):
            out.append(f"{path}: expected {expected}, got {type(value).__name__}")
            return

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        out.append(f"{path}: {value!r} is not one of {enum}")
        return

    if isinstance(value, int) and not isinstance(value, bool):
        minimum, maximum = schema.get("minimum"), schema.get("maximum")
        if minimum is not None and value < minimum:
            out.append(f"{path}: {value} below minimum {minimum}")
        if maximum is not None and value > maximum:
            out.append(f"{path}: {value} above maximum {maximum}")

    if isinstance(value, dict):
        properties = schema.get("properties") or {}
        for key in schema.get("required") or []:
            if key not in value:
                out.append(f"{path}: missing required key {key!r}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    out.append(f"{path}: unexpected key {key!r}")
        for key, sub_schema in properties.items():
            if key in value and isinstance(sub_schema, dict):
                _collect(value[key], sub_schema, f"{path}.{key}", out)

    elif isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                if len(out) >= MAX_VIOLATIONS:
                    return
                _collect(item, item_schema, f"{path}[{index}]", out)


def describe_schema(schema: dict, indent: int = 0) -> str:
    """The schema as a compact outline, for a backend that will not enforce it.

    A raw JSON Schema dump is mostly punctuation the model has to parse before it learns
    anything; this states the keys, their types and their constraints in the order they
    must be generated.
    """
    lines: list[str] = []
    _describe(schema, indent, lines)
    return "\n".join(lines)


def _describe(schema: dict, indent: int, lines: list[str]) -> None:
    pad = "  " * indent
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    for key, sub in properties.items():
        if not isinstance(sub, dict):
            continue
        kind = sub.get("type", "any")
        notes = [] if key in required else ["optional"]
        if sub.get("enum"):
            notes.append("one of " + "|".join(str(e) for e in sub["enum"]))
        if sub.get("minimum") is not None or sub.get("maximum") is not None:
            notes.append(f"{sub.get('minimum', '-∞')}..{sub.get('maximum', '∞')}")
        suffix = f"  ({', '.join(notes)})" if notes else ""
        if kind == "array":
            item = sub.get("items") or {}
            item_kind = item.get("type", "any") if isinstance(item, dict) else "any"
            lines.append(f"{pad}- {key}: array of {item_kind}{suffix}")
            if item_kind == "object":
                _describe(item, indent + 1, lines)
        else:
            lines.append(f"{pad}- {key}: {kind}{suffix}")
            if kind == "object":
                _describe(sub, indent + 1, lines)
