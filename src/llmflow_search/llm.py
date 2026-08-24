"""Ollama chat wrappers, JSON extraction, and the interactive model picker."""

import json
import re
import sys

import ollama

from .config import FAST_MODEL, FAST_MODEL_ROLES
from .console import print

_PENDING_INITIAL_TASK = ""


def pop_pending_initial_task() -> str:
    global _PENDING_INITIAL_TASK
    task = _PENDING_INITIAL_TASK
    _PENDING_INITIAL_TASK = ""
    return task


def pick_model() -> str:
    try:
        models = ollama.list()
    except Exception as e:
        print(f"[!] ollama not reachable: {e}")
        sys.exit(1)
    if not models.models:
        print("[!] No models. Run: ollama pull qwen2.5:7b")
        sys.exit(1)

    print(f"\nOllama models ({len(models.models)} total):\n")
    for i, m in enumerate(models.models, 1):
        size_gb = (m.size or 0) / (1024**3)
        print(f"  {i:>2}. {m.model:<35} {size_gb:5.1f} GB")

    available = [m.model or "" for m in models.models]

    DEFAULT_MAIN = "gemma4:26b-mlx"

    def _pick(prompt: str, default: str) -> str:
        global _PENDING_INITIAL_TASK
        default_idx = next(
            (i + 1 for i, m in enumerate(available) if m == default), None
        )
        hint = f" [Enter = {default}]" if default_idx else ""
        while True:
            try:
                choice = input(f"\n{prompt}{hint} > ").strip()
                if not choice and default_idx:
                    return default
                if choice and not choice.isdigit() and default_idx:
                    _PENDING_INITIAL_TASK = choice
                    return default
                idx = int(choice) - 1
                if 0 <= idx < len(available):
                    return available[idx]
            except (ValueError, IndexError):
                pass
            print(f"  Enter 1–{len(available)}")

    main_model = _pick("Pick model number", DEFAULT_MAIN)
    print(f"\nUsing: {main_model}")
    return main_model


_AVAILABLE_MODELS_CACHE: list[str] | None = None


def _available_models() -> list[str]:
    global _AVAILABLE_MODELS_CACHE
    if _AVAILABLE_MODELS_CACHE is None:
        try:
            _AVAILABLE_MODELS_CACHE = [m.model or "" for m in ollama.list().models]
        except Exception:
            _AVAILABLE_MODELS_CACHE = []
    return _AVAILABLE_MODELS_CACHE


def _schema_capable_model(model: str) -> str:
    """Map an MLX-tagged model to its GGUF sibling, if pulled.

    Ollama's MLX runtime does not honor ``format`` JSON-schema constrained decoding
    (only the GGUF/llama.cpp backend enforces it) — see the schema-vs-freeform
    fallback already required in nodes.py's plan-step parsing. Callers that need a
    real schema guarantee (not just a prompt request) must route through this.
    """
    if "-mlx" not in model:
        return model
    candidate = model.replace("-mlx", "")
    return candidate if candidate in _available_models() else model


def model_for_role(main_model: str, role: str) -> str:
    """The model a given decision runs on.

    Latency is the number of decisions multiplied by the latency of each, and several of
    these decisions are bookkeeping rather than judgement. With no fast model configured
    this returns the main model for every role, so the split costs nothing until it is
    turned on.
    """
    if FAST_MODEL and role in FAST_MODEL_ROLES:
        return FAST_MODEL
    return main_model


def _get_dict(msg):
    if hasattr(msg, "model_dump"):
        return msg.model_dump()
    if hasattr(msg, "dict"):
        return msg.dict()
    return msg


def _ollama_chat(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    system: str = "",
    temperature: float = 0.3,
    json_mode: bool = False,
    num_predict: int = 32768,
    format_schema: dict | None = None,
) -> dict:
    """Call ollama, normalize tool_calls to have 'id'."""
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    for m in messages:
        d = _get_dict(m)
        entry = {}
        role = d.get("role") or d.get("type", "")
        if role in ("human", "user"):
            entry["role"] = "user"
        elif role in ("ai", "assistant"):
            entry["role"] = "assistant"
        elif role == "tool":
            entry["role"] = "tool"
        elif role == "system":
            entry["role"] = "system"
        else:
            continue
        if d.get("content"):
            entry["content"] = d["content"]
        if d.get("name"):
            entry["name"] = d["name"]
        if d.get("tool_call_id"):
            entry["tool_call_id"] = d["tool_call_id"]
        # tool_calls: convert langchain flat back to ollama nested
        if d.get("tool_calls"):
            tcs = []
            for tc in d["tool_calls"]:
                if "function" in tc:
                    tcs.append(tc)
                else:
                    tcs.append(
                        {
                            "function": {
                                "name": tc.get("name", ""),
                                "arguments": tc.get("args", {}),
                            }
                        }
                    )
            entry["tool_calls"] = tcs
        msgs.append(entry)

    kwargs: dict = {
        "model": model,
        "messages": msgs,
        "tools": tools,
        "options": {"temperature": temperature, "num_predict": num_predict},
        "think": False,
    }
    if format_schema and not tools:
        # Constrained decoding: Ollama forces the output to match this JSON schema.
        kwargs["format"] = format_schema
    elif json_mode and not tools:
        kwargs["format"] = "json"
    response = ollama.chat(**kwargs)
    msg = response["message"]

    if msg.get("tool_calls"):
        import uuid as _uuid

        tcs = []
        for tc in msg["tool_calls"]:
            d = tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
            d["id"] = f"call_{_uuid.uuid4().hex[:8]}"
            tcs.append(d)
        return {"role": "assistant", "content": "", "tool_calls": tcs}
    return {"role": "assistant", "content": msg.get("content", "")}


def _extract_json_text(content: str) -> str:
    """Return the most likely JSON object/array substring from an LLM response."""
    content = content.strip()

    fenced = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    fenced = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = content.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(content)):
            char = content[idx]
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    return content[start : idx + 1]
    return content


def _retry_after_bad_json(
    model: str, messages: list[dict], system: str, content: str, **format_args
) -> str:
    try:
        json.loads(_extract_json_text(content))
        return content
    except (json.JSONDecodeError, TypeError):
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": "Your response was not valid JSON. Return valid JSON only — no explanation, no markdown.",
            },
        ]
        retry = _ollama_chat(
            model,
            retry_messages,
            tools=None,
            system=system,
            temperature=0,
            **format_args,
        )
        return retry.get("content", content)


def _ollama_chat_json(
    model: str, messages: list[dict], system: str, temperature: float = 0
) -> str:
    """Call ollama expecting JSON; retry once with explicit nudge if result is unparseable."""
    response = _ollama_chat(
        model,
        messages,
        tools=None,
        system=system,
        temperature=temperature,
        json_mode=True,
    )
    return _retry_after_bad_json(
        model, messages, system, response.get("content", ""), json_mode=True
    )


def _ollama_chat_schema(
    model: str,
    messages: list[dict],
    system: str,
    format_schema: dict,
    temperature: float = 0,
) -> str:
    """Call ollama with JSON-schema constrained decoding, on a schema-capable model.

    Routes to the GGUF sibling of an MLX-tagged model (see ``_schema_capable_model``)
    so the schema is actually enforced by the decoder rather than merely requested
    in the prompt.
    """
    schema_model = _schema_capable_model(model)
    response = _ollama_chat(
        schema_model,
        messages,
        tools=None,
        system=system,
        temperature=temperature,
        format_schema=format_schema,
    )
    return _retry_after_bad_json(
        schema_model,
        messages,
        system,
        response.get("content", ""),
        format_schema=format_schema,
    )


def _json_loads_best_effort(content: str, fallback):
    try:
        return json.loads(content)
    except (TypeError, json.JSONDecodeError):
        pass
    try:
        return json.loads(_extract_json_text(content))
    except (TypeError, json.JSONDecodeError):
        return fallback
