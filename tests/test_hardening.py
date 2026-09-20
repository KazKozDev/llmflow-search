import asyncio
import json
from types import SimpleNamespace

import pytest

from llmflow_search import llm, mcp_client, trace
from llmflow_search.memory import ResearchMemoryStore
from llmflow_search.nodes import (
    _execute_single_step,
    _normalize_evidence_ledger_result,
    verify_node,
)
from llmflow_search.profiles import FOOTNOTE_PROFILE, GENERIC_PROFILE
from llmflow_search.search_memory import _default_search_memory
from llmflow_search.sources import _sources_from_tool_result
from llmflow_search.tool_policy import ToolEffect, authorize_tool_call, classify_tool


def _source(index=1):
    return {
        "url": f"https://example.test/{index}",
        "title": f"Source {index}",
        "content": "The supported fact is present.",
        "kind": "page",
    }


def test_strict_readiness_requires_every_requirement_index():
    raw = {
        "ledger": [
            {
                "requirement_index": 0,
                "source_ids": [1],
                "support_status": "supported",
                "support_level": "supported",
                "can_use_in_answer": True,
            }
        ],
        "global_missing": [],
        "next_steps": [],
    }
    result = _normalize_evidence_ledger_result(
        raw, [_source()], ["requirement A", "requirement B"], "strict"
    )
    assert result["answer_ready"] is False


def test_ledger_rejects_source_ids_not_visible_in_prompt():
    sources = [_source(index) for index in range(1, 4)]
    raw = {
        "ledger": [
            {
                "requirement_index": 0,
                "source_ids": [3],
                "support_status": "supported",
                "support_level": "supported",
                "can_use_in_answer": True,
            }
        ],
        "global_missing": [],
        "next_steps": [],
    }
    result = _normalize_evidence_ledger_result(
        raw, sources, ["requirement"], "strict", valid_source_ids={1, 2}
    )
    assert result["admissible_sources"] == []
    assert result["answer_ready"] is False


def test_long_json_mcp_result_stays_valid_and_extractable():
    payload = json.dumps(
        {"url": "https://example.test/long", "title": "Long", "text": "x" * 25000}
    )
    result = SimpleNamespace(
        content=[SimpleNamespace(text=payload)], structuredContent=None, isError=False
    )
    bounded = mcp_client._mcp_result_text(result)
    assert isinstance(json.loads(bounded), dict)
    assert _sources_from_tool_result("web_read", bounded)


def test_multiple_json_mcp_blocks_remain_one_valid_container():
    result = SimpleNamespace(
        content=[
            SimpleNamespace(text=json.dumps({"row": 1, "text": "x" * 18000})),
            SimpleNamespace(text=json.dumps({"row": 2, "text": "y" * 18000})),
        ],
        structuredContent=None,
        isError=False,
    )
    bounded = mcp_client._mcp_result_text(result)
    assert isinstance(json.loads(bounded), list)


def _verify_state():
    return {
        "task": "prove the fact",
        "conversation_context": "",
        "requirements_result": {
            "completion_criteria": ["prove the fact"],
            "quality_preferences": [],
            "answer_mode": "strict",
        },
        "plan": [],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [_source()],
        "admissible_sources": [_source()],
        "sources": [_source()],
        "evidence_ledger_result": {},
        "evidence_challenge_result": {},
        "draft_result": {
            "answer": "Unverified draft [1].",
            "salvaged_prose": True,
            "insufficient_evidence": False,
        },
        "verification_result": {},
        "evidence_audit": {"passed": True, "gaps": []},
        "constraint_registry": [],
        "final_answer": "",
        "iteration": 0,
        "evidence_round": 0,
        "search_memory": _default_search_memory(),
        "answer_mode": "strict",
        "stagnant_rounds": 0,
        "last_supported_claim_count": 0,
    }


def test_empty_verifier_output_fails_closed(monkeypatch):
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})
    result = asyncio.run(verify_node(_verify_state(), "model", [], FOOTNOTE_PROFILE))
    assert result["verification_result"]["task_complete"] is False
    assert result["verification_result"]["insufficient_evidence"] is True


def test_malformed_or_unknown_verdict_fields_fail_closed(monkeypatch):
    monkeypatch.setattr(
        llm, "_ollama_chat", lambda *a, **k: {"content": "Grounded fact [1]."}
    )
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps(
            {"coverage_complete": True, "missing": [], "notes": [], "trusted": True}
        ),
    )
    result = asyncio.run(verify_node(_verify_state(), "model", [], FOOTNOTE_PROFILE))
    assert result["verification_result"]["task_complete"] is False


def test_tool_policy_requires_authorization_for_mutations():
    tools = [
        {
            "function": {
                "name": "publish_report",
                "description": "Publish a report",
                "parameters": {"type": "object"},
            }
        }
    ]
    assert classify_tool("publish_report", tools).effect is ToolEffect.EXTERNAL_WRITE
    with pytest.raises(PermissionError):
        asyncio.run(authorize_tool_call("publish_report", {}, tools, None))


def test_multiple_native_tool_calls_are_rejected_before_transport(monkeypatch):
    monkeypatch.setattr(
        llm,
        "_ollama_chat",
        lambda *a, **k: {
            "tool_calls": [
                {"function": {"name": "get_a", "arguments": {}}},
                {"function": {"name": "get_b", "arguments": {}}},
            ]
        },
    )
    tools = [
        {
            "function": {
                "name": name,
                "parameters": {"type": "object", "properties": {}},
            }
        }
        for name in ("get_a", "get_b")
    ]
    _step, result, sources, updates = asyncio.run(
        _execute_single_step(
            "plain instruction", "model", tools, None, "question", {}, GENERIC_PROFILE
        )
    )
    assert "exactly one is allowed" in result
    assert sources == [] and updates == {}


def test_mcp_timeout_escapes_execution_controller(monkeypatch):
    async def time_out(*args, **kwargs):
        raise TimeoutError("wedged subprocess")

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", time_out)
    tools = [
        {
            "function": {
                "name": "get_fact",
                "description": "Read a fact",
                "parameters": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            }
        }
    ]
    with pytest.raises(TimeoutError, match="wedged subprocess"):
        asyncio.run(
            _execute_single_step(
                'get_fact: {"topic": "capital of France"}',
                "model",
                tools,
                object(),
                "question",
                {},
                GENERIC_PROFILE,
            )
        )


def test_memory_file_is_private_and_atomic(tmp_path):
    path = tmp_path / "memory.json"
    store = ResearchMemoryStore(str(path))
    store.add_experience({"task": "one"})
    assert json.loads(path.read_text())["experiences"][0]["task"] == "one"
    assert path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_independent_memory_stores_merge_under_lock(tmp_path):
    path = tmp_path / "memory.json"
    first = ResearchMemoryStore(str(path))
    second = ResearchMemoryStore(str(path))
    first.add_experience({"task": "one"})
    second.add_experience({"task": "two"})
    tasks = [item["task"] for item in json.loads(path.read_text())["experiences"]]
    assert tasks == ["one", "two"]


def test_trace_context_isolated_between_concurrent_tasks(tmp_path):
    async def emit_query(name):
        path = tmp_path / f"{name}.jsonl"
        trace.start_run(path, run_id=name)
        trace.begin_query(name)
        await asyncio.sleep(0)
        trace.emit("done")
        trace.close_run()
        return json.loads(path.read_text().splitlines()[-1])

    async def run():
        return await asyncio.gather(emit_query("q1"), emit_query("q2"))

    first, second = asyncio.run(run())
    assert first["query_id"] == "q1"
    assert second["query_id"] == "q2"


def _read_view(**overrides):
    from llmflow_search.policy import RoundView

    base = dict(
        remaining=(),
        ranked_unread=("https://a.test", "https://b.test", "https://c.test"),
        known_urls=frozenset(
            {"https://a.test", "https://b.test", "https://c.test"}
        ),
        tool_just_run="web_search",
        has_read_anything=False,
    )
    base.update(overrides)
    return RoundView(**base)


def test_model_selection_leaves_the_choice_to_the_controller(monkeypatch):
    from llmflow_search import policy

    monkeypatch.setattr(policy, "READ_SELECTION", "model")
    forced, by = policy.forced_reads(_read_view(has_read_anything=True))
    assert forced == []
    assert by == ""


def test_auto_selection_still_opens_the_top_of_the_catalog(monkeypatch):
    from llmflow_search import policy

    monkeypatch.setattr(policy, "READ_SELECTION", "auto")
    monkeypatch.setattr(policy, "AUTO_READ_FOLLOWUP_TOP_K", 2)
    forced, by = policy.forced_reads(_read_view(has_read_anything=True))
    assert by == "auto_rank"
    assert len(forced) == 2


def test_a_round_that_would_open_nothing_still_opens_something(monkeypatch):
    """The failure the unconditional rule exists for: answering from snippets alone."""
    from llmflow_search import policy

    monkeypatch.setattr(policy, "READ_SELECTION", "model")
    action = policy.decide(_read_view(), decision="NEXT", proposed_steps=[])
    assert action.reads
    assert action.source == "auto_fallback"


def test_the_net_does_not_fire_once_something_has_been_read(monkeypatch):
    from llmflow_search import policy

    monkeypatch.setattr(policy, "READ_SELECTION", "model")
    action = policy.decide(
        _read_view(has_read_anything=True),
        decision="NEXT",
        proposed_steps=["web_search: another angle"],
    )
    assert not action.reads


def test_the_net_does_not_override_a_read_the_controller_chose(monkeypatch):
    from llmflow_search import policy

    monkeypatch.setattr(policy, "READ_SELECTION", "model")
    action = policy.decide(
        _read_view(),
        decision="NEXT",
        proposed_steps=["web_read: https://b.test"],
    )
    assert action.reads == ("https://b.test",)
    assert action.source == "model"
