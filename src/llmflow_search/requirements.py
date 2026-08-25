"""User-contract extraction and normalization."""

from . import llm, trace
from .console import print
from .llm import _json_loads_best_effort
from .profiles import Profile
from .sources import _effective_question
from .state import AgentState


def _default_requirements(question: str) -> dict:
    return {
        "target": question,
        "answer_mode": "strict",
        "scope": "answer the user question",
        "granularity": None,
        "unit_or_pair": None,
        "required_coverage": "all explicitly requested parts of the question",
        "output_format": "direct answer",
        "quality_preferences": [],
        "completion_criteria": [
            "The answer addresses every explicit requirement in the user question.",
            "Every factual claim is grounded in the provided sources.",
            "The answer uses one consistent unit, currency pair, or measurement basis unless the user explicitly asked for multiple.",
        ],
        "missing_data_policy": "If any required part cannot be sourced, mark task_complete=false and list the gap.",
        "search_hints": [],
    }


def _normalize_requirements(raw: dict | None, question: str) -> dict:
    defaults = _default_requirements(question)
    if not isinstance(raw, dict):
        return defaults
    result = defaults | {
        key: value
        for key, value in raw.items()
        if key in defaults and value not in (None, "")
    }
    for key in ("quality_preferences", "completion_criteria", "search_hints"):
        if not isinstance(result.get(key), list):
            result[key] = defaults[key]
        result[key] = [str(item) for item in result[key] if str(item).strip()]
    if result.get("answer_mode") not in ("strict", "roundup"):
        result["answer_mode"] = "strict"
    return result


def _proof_requirements(requirements: dict) -> dict:
    """Return the user-contract fields that can block a grounded answer."""
    return {
        "target": requirements.get("target"),
        "scope": requirements.get("scope"),
        "granularity": requirements.get("granularity"),
        "unit_or_pair": requirements.get("unit_or_pair"),
        "required_coverage": requirements.get("required_coverage"),
        "output_format": requirements.get("output_format"),
        "completion_criteria": list(requirements.get("completion_criteria", [])),
        "missing_data_policy": requirements.get("missing_data_policy"),
    }


REQUIREMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string"},
        "scope": {"type": "string"},
        "granularity": {"type": ["string", "null"]},
        "unit_or_pair": {"type": ["string", "null"]},
        "required_coverage": {"type": "string"},
        "output_format": {"type": "string"},
        "quality_preferences": {"type": "array", "items": {"type": "string"}},
        "completion_criteria": {"type": "array", "items": {"type": "string"}},
        "missing_data_policy": {"type": "string"},
        "search_hints": {"type": "array", "items": {"type": "string"}},
        "answer_mode": {"type": "string", "enum": ["strict", "roundup"]},
    },
    "required": ["target", "scope", "completion_criteria", "answer_mode"],
}


async def requirements_node(
    state: AgentState, model: str, _tools: list[dict], profile: Profile
) -> dict:
    """Extract task completion requirements before planning."""
    if state.get("requirements_result"):
        return {"iteration": state["iteration"] + 1}

    question = _effective_question(state["task"])
    iteration = state["iteration"] + 1
    print("\n  [REQUIREMENTS] Extracting completion criteria...")
    requirements_input = (
        f"QUESTION:\n{question}\n\nExtract task completion requirements."
    )
    role_model = llm.model_for_role(model, "requirements")
    with trace.model_call("requirements", requirements_input, role_model):
        content = llm._ollama_chat_schema(
            role_model,
            [{"role": "user", "content": requirements_input}],
            system=profile.requirements,
            format_schema=REQUIREMENTS_SCHEMA,
        )
    raw = _json_loads_best_effort(content, {})
    requirements = _normalize_requirements(raw, question)
    criteria = requirements.get("completion_criteria", [])
    print(
        f"  [REQUIREMENTS] {len(criteria)} criteria, answer_mode={requirements.get('answer_mode')}"
    )
    for i, criterion in enumerate(criteria[:4], 1):
        print(f"    {i}. {criterion[:100]}")
    return {"requirements_result": requirements, "iteration": iteration}
