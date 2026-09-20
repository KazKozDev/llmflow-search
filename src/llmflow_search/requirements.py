"""User-contract extraction and normalization."""

from . import llm, trace
from .config import SEPARATE_IDENTIFYING_CRITERIA
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
        # Positions in completion_criteria that only identify the subject. Empty by
        # default: with no marking, every criterion is treated as reportable, which is
        # the behaviour that shipped before this field existed.
        "identifying_criteria": [],
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
    result["identifying_criteria"] = _identifying_indices(
        raw.get("identifying_criteria"), len(result["completion_criteria"])
    )
    return result


def _identifying_indices(raw, criterion_count: int) -> list[int]:
    """Which criteria are clues for finding the subject rather than things to report.

    A puzzle question — born in 1886, mistaken for a shaman on a 1915 trip, 35 years in
    one house — decomposes into a conjunction, and the decomposition is right: each part
    has to be searched for separately. What was wrong was treating every part as something
    the answer must *state, with a citation*. Those clues exist to single out one entity;
    the user wants the entity, not a sourced restatement of the clues they supplied.
    Requiring proof of all of them makes strict mode unreachable on exactly the questions
    it was built for, and a run refuses while already holding the answer.

    Out-of-range and duplicate positions are dropped rather than repaired: an index this
    function cannot place is a criterion nobody can tell apart, and the safe reading of an
    unplaceable marking is that the criterion is reportable.
    """
    if not SEPARATE_IDENTIFYING_CRITERIA or not isinstance(raw, list):
        return []
    indices: list[int] = []
    for item in raw:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < criterion_count and index not in indices:
            indices.append(index)
    # A question whose every criterion is a clue still has to prove something before it
    # may answer. Marking all of them would make the gate vacuous, so it is refused whole.
    if len(indices) >= criterion_count:
        return []
    return sorted(indices)


def reportable_indices(requirements: dict) -> set[int]:
    """The criteria that must be supported before an answer is permitted."""
    criteria = list(requirements.get("completion_criteria", []))
    identifying = set(requirements.get("identifying_criteria") or [])
    return {index for index in range(len(criteria)) if index not in identifying}


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
        "identifying_criteria": list(requirements.get("identifying_criteria") or []),
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
        "identifying_criteria": {"type": "array", "items": {"type": "integer"}},
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
    identifying = set(requirements.get("identifying_criteria") or [])
    if identifying:
        print(
            f"  [REQUIREMENTS] {len(criteria) - len(identifying)} to prove,"
            f" {len(identifying)} identifying clues"
        )
    for i, criterion in enumerate(criteria[:4], 1):
        mark = "clue " if (i - 1) in identifying else "prove"
        print(f"    {i}. [{mark}] {criterion[:95]}")
    return {"requirements_result": requirements, "iteration": iteration}
