from __future__ import annotations

import asyncio
import json

from llmflow_search import agent as agent_module
from llmflow_search import llm, mcp_client, memory
from llmflow_search import nodes as nodes_module
from llmflow_search.config import INSUFFICIENT_EVIDENCE_MESSAGE
from llmflow_search.nodes import (
    _normalize_evidence_ledger_result,
    evidence_ledger_node,
    execute_node,
    route_after_evidence_challenge,
    route_after_evidence_ledger,
    route_after_evidence_reextract,
    route_after_plan,
    strategy_node,
)
from llmflow_search.profiles import FOOTNOTE_PROFILE, GENERIC_PROFILE
from llmflow_search.prompts import (
    ANSWER_PROSE_SYSTEM_PROMPT,
    EVIDENCE_CHALLENGE_SYSTEM_PROMPT,
    EVIDENCE_LEDGER_SYSTEM_PROMPT,
    OBSERVATION_SYSTEM_PROMPT,
)
from llmflow_search.tool_steps import _tool_call_from_schema_step


def test_agent_graph_completes_with_fake_model_and_fake_tool(monkeypatch):
    def fake_chat(
        model,
        messages,
        tools=None,
        system="",
        temperature=0.3,
        json_mode=False,
        num_predict=32768,
        format_schema=None,
    ):
        if system == agent_module.REQUIREMENTS_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "target": "test rate",
                        "scope": "single fact",
                        "granularity": None,
                        "unit_or_pair": "EUR/RUB",
                        "required_coverage": "rate value",
                        "output_format": "sentence",
                        "completion_criteria": ["Provide the sourced rate."],
                        "missing_data_policy": "fail if missing",
                        "search_hints": [],
                    }
                )
            }
        if system == agent_module.PLAN_PROMPT:
            return {"content": json.dumps(["web_read: https://example.com/source"])}
        # Prose-first answer/verify flow: draft prose, re-ground it, then a compact
        # JSON verdict (see answer_node / verify_node).
        if system == agent_module.ANSWER_PROSE_SYSTEM_PROMPT:
            return {"content": "The sourced EUR/RUB rate is 90.1 [1]."}
        if system == agent_module.VERIFY_PROSE_SYSTEM_PROMPT:
            return {"content": "The sourced EUR/RUB rate is 90.1 [1]."}
        if system == agent_module.VERIFY_VERDICT_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {"coverage_complete": True, "missing": [], "notes": []}
                )
            }
        if system == agent_module.EVIDENCE_LEDGER_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "answer_ready": True,
                        "ledger": [
                            {
                                "requirement": "rate value",
                                "proposed_claim": "The sourced EUR/RUB rate is 90.1.",
                                "source_ids": [1],
                                "support_level": "supported",
                                "can_use_in_answer": True,
                                "missing": "",
                            }
                        ],
                        "global_missing": [],
                        "next_steps": [],
                        "reason": "The fetched source supports the required rate.",
                    }
                )
            }
        if system == agent_module.EVIDENCE_CHALLENGE_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "answer_permitted": True,
                        "blocking_gaps": [],
                        "next_steps": [],
                        "reason": "No blocking evidence weakness.",
                    }
                )
            }
        # execute_node's post-batch controller decides DONE/CONTINUE/NEXT. Its prompt is
        # a local string, so match on stable wording. The web_read source is fetched, so
        # DONE is valid and ends the loop.
        if "After executing a batch of steps" in system:
            return {
                "content": json.dumps({"decision": "DONE", "reason": "rate is covered"})
            }
        raise AssertionError(f"unexpected system prompt: {system[:80]}")

    async def fake_call_mcp_tool(name, args, session=None):
        assert name == "web_read"
        return json.dumps(
            {
                "url": args["url"],
                "title": "Source",
                "text": "Official source says EUR/RUB rate is 90.1.",
                "pub_date": "2026-05-01",
                "text_length": 41,
            }
        )

    class FakeStore:
        def get_strategies(self, limit=5):
            return []

        def get_skills(self, limit=5):
            return []

        def best_strategy(self):
            return None

        def record_strategy(self, *args, **kwargs):
            return None

        def save_skill(self, *args, **kwargs):
            return None

        def add_experience(self, *args, **kwargs):
            return None

    # The volatile IO functions live in their own modules and are called module-qualified,
    # so patch them where they are defined (not on the re-export facade).
    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(memory, "_get_research_store", lambda: FakeStore())

    graph = agent_module.build_graph("fake-model", tools=[])
    state = {
        "task": "Find the test rate at https://example.com/source",
        "requirements_result": {},
        "plan": [],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "evidence_ledger_result": {},
        "evidence_challenge_result": {},
        "draft_result": {},
        "verification_result": {},
        "evidence_audit": {},
        "final_answer": "",
        "iteration": 0,
        "replan_count": 0,
        "evidence_round": 0,
        "search_memory": agent_module._default_search_memory(),
    }

    final = asyncio.run(graph.ainvoke(state, {"recursion_limit": 80}))

    assert final["final_answer"] == "The sourced EUR/RUB rate is 90.1 [1]."
    assert final["verification_result"]["task_complete"] is True
    assert len(final["completed_steps"]) == 1
    assert final["sources"][0]["url"] == "https://example.com/source"


def test_evidence_challenge_can_force_another_tool_step(monkeypatch):
    ledger_calls = 0
    challenge_calls = 0

    def fake_chat(
        model,
        messages,
        tools=None,
        system="",
        temperature=0.3,
        json_mode=False,
        num_predict=32768,
        format_schema=None,
    ):
        nonlocal ledger_calls, challenge_calls
        if system == agent_module.REQUIREMENTS_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "target": "test fact",
                        "scope": "single fact",
                        "completion_criteria": ["Provide the sourced fact."],
                    }
                )
            }
        if system == agent_module.PLAN_PROMPT:
            return {"content": json.dumps(["web_read: https://example.com/weak"])}
        if system == agent_module.EVIDENCE_LEDGER_SYSTEM_PROMPT:
            ledger_calls += 1
            source_id = 1 if ledger_calls == 1 else 2
            return {
                "content": json.dumps(
                    {
                        "answer_ready": True,
                        "ledger": [
                            {
                                "requirement": "test fact",
                                "proposed_claim": "The verified fact is 42.",
                                "source_ids": [source_id],
                                "support_level": "supported",
                                "can_use_in_answer": True,
                                "missing": "",
                            }
                        ],
                        "global_missing": [],
                        "next_steps": [],
                        "reason": "Ledger claims support.",
                    }
                )
            }
        if system == agent_module.EVIDENCE_CHALLENGE_SYSTEM_PROMPT:
            challenge_calls += 1
            if challenge_calls == 1:
                return {
                    "content": json.dumps(
                        {
                            "answer_permitted": False,
                            "blocking_gaps": [
                                {
                                    "requirement_index": 0,
                                    "text": "The first source is too thin for the claim.",
                                }
                            ],
                            "next_steps": ["web_read: https://example.com/strong"],
                            "reason": "Need a stronger source.",
                        }
                    )
                }
            return {
                "content": json.dumps(
                    {
                        "answer_permitted": True,
                        "blocking_gaps": [],
                        "next_steps": [],
                        "reason": "The stronger source resolves the gap.",
                    }
                )
            }
        if system == agent_module.ANSWER_PROSE_SYSTEM_PROMPT:
            return {"content": "The verified fact is 42 [1]."}
        if system == agent_module.VERIFY_PROSE_SYSTEM_PROMPT:
            return {"content": "The verified fact is 42 [1]."}
        if system == agent_module.VERIFY_VERDICT_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {"coverage_complete": True, "missing": [], "notes": []}
                )
            }
        if "After executing a batch of steps" in system:
            return {
                "content": json.dumps(
                    {"decision": "DONE", "reason": "continue to ledger"}
                )
            }
        raise AssertionError(f"unexpected system prompt: {system[:80]}")

    async def fake_call_mcp_tool(name, args, session=None):
        assert name == "web_read"
        if args["url"].endswith("/weak"):
            return json.dumps(
                {
                    "url": args["url"],
                    "title": "Weak",
                    "text": "A short mention says 42.",
                }
            )
        return json.dumps(
            {"url": args["url"], "title": "Strong", "text": "The verified fact is 42."}
        )

    class FakeStore:
        def best_strategy(self):
            return None

        def record_strategy(self, *a, **k):
            return None

        def save_skill(self, *a, **k):
            return None

        def add_experience(self, *a, **k):
            return None

    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(memory, "_get_research_store", lambda: FakeStore())

    graph = agent_module.build_graph("fake-model", tools=[])
    state = {
        "task": "Find the test fact at https://example.com/weak and https://example.com/strong",
        "requirements_result": {},
        "plan": [],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "evidence_ledger_result": {},
        "evidence_challenge_result": {},
        "draft_result": {},
        "verification_result": {},
        "evidence_audit": {},
        "final_answer": "",
        "iteration": 0,
        "replan_count": 0,
        "evidence_round": 0,
        "search_memory": agent_module._default_search_memory(),
    }

    final = asyncio.run(graph.ainvoke(state, {"recursion_limit": 120}))

    assert [step["step"] for step in final["completed_steps"]] == [
        "web_read: https://example.com/weak",
        "web_read: https://example.com/strong",
    ]
    assert final["final_answer"] == "The verified fact is 42 [1]."
    assert final["sources"][0]["url"] == "https://example.com/strong"
    assert challenge_calls == 2


def test_evidence_contract_does_not_require_unasked_exhaustiveness():
    assert "Do not invent new requirements" in EVIDENCE_CHALLENGE_SYSTEM_PROMPT
    assert (
        "unless the user explicitly requested exhaustive coverage"
        in EVIDENCE_CHALLENGE_SYSTEM_PROMPT
    )
    assert "must not appear in blocking_gaps" in EVIDENCE_CHALLENGE_SYSTEM_PROMPT
    assert "must not create ledger gaps" in EVIDENCE_LEDGER_SYSTEM_PROMPT
    assert "Be exhaustive" not in ANSWER_PROSE_SYSTEM_PROMPT
    assert "Default to completeness over brevity" not in ANSWER_PROSE_SYSTEM_PROMPT


def test_proof_contract_excludes_non_blocking_preferences():
    requirements = nodes_module._normalize_requirements(
        {
            "target": "requested information",
            "scope": "answer the request",
            "required_coverage": "cover the requested topic",
            "completion_criteria": ["Answer the explicit request."],
            "quality_preferences": ["Prefer variety when supported."],
            "search_hints": ["optional discovery hint"],
        },
        "requested information",
    )

    proof = nodes_module._proof_requirements(requirements)

    assert proof["completion_criteria"] == ["Answer the explicit request."]
    assert "quality_preferences" not in proof
    assert "search_hints" not in proof
    assert requirements["quality_preferences"] == ["Prefer variety when supported."]


def test_evidence_ledger_keeps_stable_claim_metadata():
    sources = [
        {
            "title": "Article",
            "url": "https://example.org/article",
            "content": "The event happened on 2026-06-30 in Girona.",
        }
    ]

    ledger = _normalize_evidence_ledger_result(
        {
            "answer_ready": True,
            "ledger": [
                {
                    "claim_id": "girona-event",
                    "requirement": "bounded news selection",
                    "proposed_claim": "The event happened in Girona.",
                    "event_date": "2026-06-30",
                    "publication_date": "2026-07-01",
                    "location": "Girona",
                    "source_ids": [1],
                    "source_quality": "secondary",
                    "support_status": "supported",
                    "support_level": "supported",
                    "can_use_in_answer": True,
                    "missing": "",
                    "rejection_reason": "",
                }
            ],
            "global_missing": [],
            "next_steps": [],
        },
        sources,
    )

    assert ledger["answer_ready"] is True
    assert ledger["supported_claim_count"] == 1
    assert ledger["ledger"][0]["claim_id"] == "girona-event"
    assert ledger["ledger"][0]["event_date"] == "2026-06-30"
    assert ledger["ledger"][0]["location"] == "Girona"
    assert ledger["ledger"][0]["support_status"] == "supported"
    assert ledger["admissible_sources"] == sources


def test_terminal_insufficient_evidence_does_not_reenter_strategy():
    state = {
        "task": "Find the requested fact",
        "plan": [],
        "completed_steps": [{"step": "web_search: requested fact"}],
        "evidence_audit": {
            "passed": False,
            "gaps": ["No fresh evidence path remains."],
        },
        "verification_result": {"insufficient_evidence": True},
        "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "evidence_round": 1,
    }

    assert route_after_evidence_challenge(state) == "assimilate"


def test_terminal_insufficient_evidence_routes_supported_sources_to_partial_answer():
    state = {
        "task": "Return every requested item",
        # This reproduces the original blank-output bug: the challenge found a fresh
        # next step, but the executor cannot run it because the 40-step cap is reached.
        "plan": ["web_search: one more date check"],
        "completed_steps": [{"step": f"web_search: query {i}"} for i in range(40)],
        "sources": [
            {
                "title": "Supported source",
                "url": "https://example.com/source",
                "content": "Supported.",
            }
        ],
        "evidence_audit": {
            "passed": False,
            "gaps": ["Some requested items remain unverified."],
        },
        "verification_result": {"insufficient_evidence": True},
        "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "evidence_round": 1,
    }

    assert route_after_evidence_challenge(state) == "answer"


def test_answer_node_requests_explicit_supported_partial_answer(monkeypatch):
    captured_prompt = ""

    def fake_chat(model, messages, **kwargs):
        nonlocal captured_prompt
        captured_prompt = messages[0]["content"]
        return {"content": "Partial answer: one supported finding [1]."}

    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
    state = {
        "task": "Return every requested item",
        "requirements_result": {
            "completion_criteria": ["Include every requested item."],
            "quality_preferences": [],
        },
        "sources": [
            {
                "title": "Source",
                "url": "https://example.com/source",
                "content": "One finding.",
            }
        ],
        "evidence_audit": {
            "passed": False,
            "gaps": ["Some requested items remain unverified."],
        },
        "iteration": 0,
    }

    update = asyncio.run(
        nodes_module.answer_node(state, "fake-model", [], FOOTNOTE_PROFILE)
    )

    assert update["draft_result"]["answer"].startswith("Partial answer:")
    assert update["draft_result"]["insufficient_evidence"] is False
    assert "PARTIAL_ANSWER_MODE:\ntrue" in captured_prompt
    assert "Some requested items remain unverified." in captured_prompt
    assert "never fill a gap with outside knowledge" in captured_prompt


def test_agent_writes_debug_report_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("LLMFLOW_SEARCH_DEBUG_REPORTS", "1")
    monkeypatch.setenv("LLMFLOW_SEARCH_DEBUG_REPORT_DIR", str(tmp_path))
    state = {
        "task": "Debug task",
        "requirements_result": {"target": "debug"},
        "search_memory": {"attempted_queries": ["q"]},
        "sources": [{"url": "https://example.com", "title": "S", "content": "abc"}],
        "verification_result": {"task_complete": False, "gaps": ["gap"]},
    }

    path = agent_module._write_debug_report(state)

    assert path is not None
    data = json.loads((tmp_path / path.split("/")[-1]).read_text())
    assert data["task"] == "Debug task"
    assert data["attempted_queries"] == ["q"]


def test_agent_normalizes_python_like_tool_steps():
    call = agent_module._tool_call_from_step(
        "check_date_completeness(start_date='2026-05-01', end_date='2026-05-31', "
        "actual_items=None, granularity='day', calendar='calendar', holidays=None)"
    )

    assert call == {
        "function": {
            "name": "check_date_completeness",
            "arguments": {
                "start_date": "2026-05-01",
                "end_date": "2026-05-31",
                "actual_items": [],
                "granularity": "day",
                "calendar": "calendar",
                "holidays": [],
            },
        }
    }


def test_agent_accepts_direct_json_fetch_steps():
    call = agent_module._tool_call_from_step(
        "web_fetch_json: https://example.com/api.json"
    )

    assert call == {
        "function": {
            "name": "web_fetch_json",
            "arguments": {"url": "https://example.com/api.json"},
        }
    }


def test_live_schema_resolves_previously_unhardcoded_single_argument_tool():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "papers_search",
                "description": "Search papers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    call = _tool_call_from_schema_step(
        "papers_search: retrieval augmented generation", tools
    )

    assert call == {
        "function": {
            "name": "papers_search",
            "arguments": {"query": "retrieval augmented generation"},
        }
    }


def test_live_schema_resolves_multi_argument_tool_from_json_without_hardcoding():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "corroborate_claim",
                "description": "Corroborate a claim.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "excerpts": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["claim", "excerpts"],
                },
            },
        }
    ]

    call = _tool_call_from_schema_step(
        'corroborate_claim: {"claim":"GDP grew","excerpts":["Source A","Source B"]}',
        tools,
    )

    assert call["function"]["name"] == "corroborate_claim"
    assert call["function"]["arguments"]["excerpts"] == ["Source A", "Source B"]
    assert _tool_call_from_schema_step("corroborate_claim: GDP grew", tools) is None


def test_plan_step_objects_preserve_structured_arguments_as_json():
    steps = nodes_module._steps_from_plan_objects(
        json.dumps(
            {
                "steps": [
                    {
                        "tool": "export_dataset",
                        "arguments": {
                            "rows": [{"date": "2026-08-05"}],
                            "format": "csv",
                        },
                    }
                ]
            }
        )
    )

    assert steps == ['export_dataset: {"rows":[{"date":"2026-08-05"}],"format":"csv"}']


def test_planner_receives_catalog_for_every_live_mcp_tool(monkeypatch):
    captured = ""

    def fake_chat(model, messages, **kwargs):
        nonlocal captured
        captured = messages[0]["content"]
        return {
            "content": json.dumps({"steps": [{"tool": "web_screenshot", "arg": "{}"}]})
        }

    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "papers_search",
                "description": "Search papers.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_screenshot",
                "description": "Capture a screenshot.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    state = {
        "task": "Capture the current browser page",
        "conversation_context": "",
        "requirements_result": {},
        "plan": [],
        "iteration": 0,
        "evidence_round": 0,
        "search_memory": agent_module._default_search_memory(),
    }

    update = asyncio.run(
        nodes_module.plan_node(state, "fake-model", tools, GENERIC_PROFILE)
    )

    assert "papers_search(query*:string)" in captured
    assert "web_screenshot()" in captured
    assert update["plan"] == ["web_screenshot: {}"]


def test_strategy_plan_uses_memory_queries_as_search_steps():
    memory = agent_module._default_search_memory()
    memory["next_queries"] = ["specific source query", "another query"]

    steps = agent_module._strategy_plan_from_memory(
        "daily rate for requested range", memory
    )

    # The refactored strategy emits search-only steps; the post-batch LLM decides
    # which result URLs to read. No deterministic placeholder scaffolding remains.
    assert steps == ["web_search: specific source query", "web_search: another query"]
    assert all(step.startswith("web_search: ") for step in steps)


def test_strategy_plan_does_not_fall_back_to_question_when_no_memory_queries():
    steps = agent_module._strategy_plan_from_memory(
        "daily rate", agent_module._default_search_memory()
    )

    assert steps == []


def test_execute_dedups_web_search_batch_and_runs_sequentially(monkeypatch):
    calls = []

    async def fake_call_mcp_tool(name, args, session=None):
        calls.append((name, args["query"]))
        return json.dumps({"count": 0, "sources": []})

    def fake_ollama_chat_json(*args, **kwargs):
        return json.dumps({"decision": "DONE", "reason": "batch behavior covered"})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(llm, "_ollama_chat_schema", fake_ollama_chat_json)
    state = {
        "task": "Find the requested fact",
        "requirements_result": {},
        "plan": [
            "web_search: repeated query",
            "web_search: repeated query",
            "web_search: different query",
        ],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "search_memory": agent_module._default_search_memory(),
        "iteration": 0,
    }

    update = asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE))

    assert calls == [
        ("web_search", "repeated query"),
        ("web_search", "different query"),
    ]
    assert [step["step"] for step in update["completed_steps"]] == [
        "web_search: repeated query",
        "web_search: different query",
    ]


def test_strategy_exhausts_when_no_fresh_search_direction(monkeypatch):
    def fake_chat(
        model,
        messages,
        tools=None,
        system="",
        temperature=0.3,
        json_mode=False,
        num_predict=32768,
        format_schema=None,
    ):
        return {
            "content": json.dumps(
                {
                    "search_hypothesis": "Try a different evidence path.",
                    "failure_diagnosis": "Prior searches did not produce usable article-level evidence.",
                    "mutation_dimension": "stop",
                    "exhausted_direction": "general web search",
                    "next_queries": [],
                    "strategy_note": "No fresh search direction remains.",
                }
            )
        }

    class FakeStore:
        def best_strategy(self):
            return None

        def get_strategies(self, limit=5):
            return []

        def get_skills(self, limit=5):
            return []

    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
    monkeypatch.setattr(memory, "_get_research_store", lambda: FakeStore())
    search_memory = agent_module._default_search_memory()
    search_memory["attempted_queries"] = ["requested fact"]

    state = {
        "task": "Find the requested fact",
        "requirements_result": {
            "target": "requested fact",
            "completion_criteria": ["Provide one sourced answer."],
        },
        "plan": [],
        "completed_steps": [{"step": "web_search: requested fact"}],
        "scratchpad": "",
        "sources": [],
        "verification_result": {"insufficient_evidence": True},
        "evidence_audit": {"passed": False, "gaps": ["No supported evidence."]},
        "final_answer": "",
        "iteration": 0,
        "evidence_round": 1,
        "search_memory": search_memory,
    }

    update = asyncio.run(strategy_node(state, "fake", [], FOOTNOTE_PROFILE))

    updated_memory = update["search_memory"]
    assert updated_memory["search_exhausted"] is True
    assert updated_memory["next_queries"] == []
    assert updated_memory["search_hypotheses"] == ["Try a different evidence path."]
    assert updated_memory["failure_diagnoses"] == [
        "Prior searches did not produce usable article-level evidence."
    ]
    assert updated_memory["mutation_history"] == ["stop"]
    assert updated_memory["exhausted_directions"] == ["general web search"]
    assert update["final_answer"] == INSUFFICIENT_EVIDENCE_MESSAGE
    assert route_after_plan(state | update) == "assimilate"


def test_generate_search_queries_step_accepts_json_arguments():
    call = agent_module._tool_call_from_step(
        'generate_search_queries: {"task":"rates","requirements":{"granularity":"day"},"max_queries":6}'
    )

    assert call == {
        "function": {
            "name": "generate_search_queries",
            "arguments": {
                "task": "rates",
                "requirements": {"granularity": "day"},
                "max_queries": 6,
            },
        }
    }


def test_recipe_run_step_and_result_become_source():
    code = "def extract(source_text, input_payload):\n    return {'rows': []}"
    step = agent_module._tool_call_from_step(
        'tool_code_run_sandboxed: {"code": '
        + json.dumps(code)
        + ', "source_text": "2026-05-01 90.1", "input_payload": {"source_url": "https://example.com/rates"}}'
    )
    sources = agent_module._sources_from_tool_result(
        "tool_code_run_sandboxed",
        json.dumps(
            {
                "ok": True,
                "result": {
                    "rows": [
                        {
                            "date": "2026-05-01",
                            "value": "90.1",
                            "unit": "EUR/USD",
                            "source_url": "https://example.com/rates",
                        }
                    ],
                    "row_count": 1,
                },
            }
        ),
    )

    assert step["function"]["name"] == "tool_code_run_sandboxed"
    assert (
        step["function"]["arguments"]["input_payload"]["source_url"]
        == "https://example.com/rates"
    )
    assert sources[0]["kind"] == "recipe_rows"
    assert sources[0]["url"] == "https://example.com/rates"


def test_observation_normalization_accepts_only_controller_tags():
    payload = {"url": "https://example.org/source", "text": "Fetched text."}
    observation = agent_module._normalize_observation(
        {
            "useful": True,
            "structured": False,
            "has_rows": False,
            "dated": False,
            "source_quality": "secondary",
            "summary": "The source partially discusses the requested item.",
            "source_title": "Partial source",
            "publication_dates": ["2026-07-01"],
            "event_dates": ["2026-06-30"],
            "urls": ["https://example.org/source#section"],
            "errors": ["paywall notice"],
            "failure_diagnosis": "The source does not prove the required detail.",
            "gaps": ["needs another source"],
            "next_action_tags": ["search_better_sources", "unsupported_custom_tag"],
            "query_candidates": [
                "specific follow-up query",
                "specific follow-up query",
            ],
            "reason": "The source only covers part of the requirement.",
        },
        "web_read",
        {"url": payload["url"]},
        payload,
        "web_read: https://example.org/source",
    )

    assert observation["useful"] is True
    assert observation["source_type"] == "secondary"
    assert observation["next_action_tags"] == ["search_better_sources"]
    assert observation["suggested_queries"] == ["specific follow-up query"]
    assert observation["query_candidates"] == ["specific follow-up query"]
    assert (
        observation["summary"] == "The source partially discusses the requested item."
    )
    assert observation["source_title"] == "Partial source"
    assert observation["publication_dates"] == ["2026-07-01"]
    assert observation["event_dates"] == ["2026-06-30"]
    assert observation["urls"] == ["https://example.org/source"]
    assert observation["errors"] == ["paywall notice"]
    assert (
        observation["failure_diagnosis"]
        == "The source does not prove the required detail."
    )


def test_evidence_audit_blocks_plan_exhaustion_without_sources(monkeypatch):
    # When the audit fails, evaluate_node asks the model to reflect; stub it out.
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})
    state = {
        "task": "Find the requested fact",
        "requirements_result": {
            "target": "requested fact",
            "required_coverage": "Provide the requested fact from fetched evidence.",
            "completion_criteria": ["Provide one sourced answer."],
        },
        "plan": [],
        "completed_steps": [],
        "scratchpad": "",
        "sources": [],
        "draft_result": {},
        "verification_result": {},
        "evidence_audit": {},
        "final_answer": "",
        "iteration": 0,
        "replan_count": 0,
        "evidence_round": 0,
        "search_memory": agent_module._default_search_memory(),
    }

    update = asyncio.run(
        agent_module.evaluate_node(state, "fake", [], FOOTNOTE_PROFILE)
    )
    routed_state = state | update

    assert update["evidence_audit"]["passed"] is False
    assert update["verification_result"]["insufficient_evidence"] is True
    assert agent_module.route_after_evaluate(routed_state) == "strategy"


def test_evidence_audit_fails_when_ledger_has_no_supported_sources():
    state = {
        "task": "euro rate for everyday may 2026",
        "requirements_result": {
            "target": "daily exchange rate",
            "granularity": "day",
            "required_coverage": "Provide a rate for each day in the requested date range.",
            "completion_criteria": ["Provide a rate for each of the 31 days."],
        },
        "candidate_sources": [
            {
                "title": "Exchange-rate article",
                "url": "https://example.org/rates-note",
                "published": "2026-05-01",
                "content": "2026-05-01 rate was 90.1.",
                "kind": "page",
            }
        ],
        "admissible_sources": [],
        "sources": [],
        "evidence_ledger_result": {
            "candidate_count": 1,
            "admissible_count": 0,
            "answer_ready": False,
            "ledger": [
                {
                    "requirement": "31 days",
                    "source_ids": [],
                    "support_level": "missing",
                    "can_use_in_answer": False,
                    "missing": "Only one day is covered.",
                }
            ],
        },
    }

    audit = agent_module._audit_evidence_state(state)

    assert audit["passed"] is False
    assert audit["gaps"] == ["No fetched sources support the evidence ledger."]
    assert audit["source_count"] == 0
    assert audit["candidate_source_count"] == 1


def test_record_observation_tracks_unhelpful_source_in_memory():
    requirements = {
        "target": "requested fact",
        "required_coverage": "Answer all requested parts.",
        "completion_criteria": ["Cover the requested fact."],
    }
    observation = agent_module._normalize_observation(
        {
            "useful": False,
            "structured": False,
            "has_rows": False,
            "dated": False,
            "source_quality": "unknown",
            "summary": "Partial source without the required detail.",
            "source_title": "Partial source",
            "failure_diagnosis": "The source lacks the required detail.",
            "gaps": ["source does not cover required detail"],
            "next_action_tags": ["search_better_sources"],
            "query_candidates": ["specific required detail source"],
            "reason": "Needs a better-matching source.",
        },
        "web_read",
        {"url": "https://example.org/source"},
        {"url": "https://example.org/source", "text": "Partial source."},
        "web_read: https://example.org/source",
    )
    memory = agent_module._record_observation(
        agent_module._default_search_memory(), observation, requirements
    )

    # Next-step planning is now an LLM decision; the deterministic memory bookkeeping
    # is what we assert: controller tags recorded, unhelpful source quarantined.
    assert observation["useful"] is False
    assert "search_better_sources" in memory["next_actions"]
    assert "source does not cover required detail" in memory["open_gaps"]
    assert "https://example.org/source" in memory["avoid_urls"]
    assert "example.org" in memory["bad_domains"]
    assert memory["query_candidates"] == ["specific required detail source"]
    assert memory["next_queries"] == []
    assert memory["observation_failure_diagnoses"] == [
        "The source lacks the required detail."
    ]
    assert (
        memory["observations"][0]["summary"]
        == "Partial source without the required detail."
    )


def test_fast_observation_enriches_candidate_source_metadata():
    sources = [
        {
            "title": "https://example.org/source",
            "url": "https://example.org/source",
            "published": None,
            "content": "Fetched text.",
            "kind": "page",
        }
    ]
    observation = {
        "source_type": "primary",
        "source_title": "Official source",
        "publication_dates": ["2026-07-01"],
        "event_dates": ["2026-06-30"],
        "summary": "Official page confirms the relevant fact.",
        "errors": [],
    }

    enriched = agent_module._enrich_sources_with_observation(sources, observation)

    assert enriched[0]["title"] == "Official source"
    assert enriched[0]["published"] == "2026-07-01"
    assert enriched[0]["source_quality"]["source_type"] == "primary"
    assert (
        enriched[0]["tool_observation_summary"]
        == "Official page confirms the relevant fact."
    )
    assert enriched[0]["event_dates"] == ["2026-06-30"]


def test_fallback_observation_records_objective_facts_for_empty_tables():
    requirements = {
        "target": "daily exchange rate",
        "granularity": "day",
        "required_coverage": "Provide a rate for each day in the requested date range.",
        "completion_criteria": ["Provide a rate for each of the 31 days."],
    }
    payload = {"url": "https://example.org/rates", "tables": [], "table_count": 0}

    observation = agent_module._fallback_observation(
        "web_extract_tables",
        {"url": payload["url"]},
        payload,
        f"web_extract_tables: {payload['url']}",
    )
    memory = agent_module._record_observation(
        agent_module._default_search_memory(), observation, requirements
    )

    # The fallback now records only objective facts and leaves strategy (gaps,
    # next actions, escalation) to the LLM — no deterministic inference.
    assert observation["structured"] is True
    assert observation["has_rows"] is False
    assert observation["useful"] is False
    assert observation["gaps"] == []
    # An unhelpful structured fetch is quarantined so the agent stops retrying it.
    assert "https://example.org/rates" in memory["avoid_urls"]


def test_generic_profile_runs_graph_against_a_foreign_tool(monkeypatch):
    """End-to-end on the GENERIC profile: the agent drives an arbitrary MCP tool via native
    tool-calling, treats its output as a source, and returns a grounded, verified answer.
    """
    from llmflow_search import prompts

    def fake_chat(
        model,
        messages,
        tools=None,
        system="",
        temperature=0.3,
        json_mode=False,
        num_predict=32768,
        format_schema=None,
    ):
        if system == prompts.REQUIREMENTS_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "target": "capital of France",
                        "required_coverage": "state the capital",
                        "completion_criteria": [
                            "Name the capital of France from a tool result."
                        ],
                    }
                )
            }
        if system == prompts.GENERIC_PLAN_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "steps": [
                            {"tool": "step", "arg": "look up the capital of France"}
                        ]
                    }
                )
            }
        if tools:  # execute step → native function call against the foreign tool
            return {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_fact",
                            "arguments": {"topic": "capital of france"},
                        }
                    }
                ],
            }
        if system == prompts.OBSERVATION_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "useful": True,
                        "structured": False,
                        "has_rows": False,
                        "dated": False,
                        "source_quality": "secondary",
                        "gaps": [],
                        "next_action_tags": ["stop_and_answer"],
                        "suggested_queries": [],
                        "reason": "answered",
                    }
                )
            }
        if system == prompts.GENERIC_POST_BATCH_PROMPT:
            return {
                "content": json.dumps(
                    {"decision": "DONE", "next_steps": [], "reason": "covered"}
                )
            }
        if system == prompts.ANSWER_PROSE_SYSTEM_PROMPT:
            return {"content": "The capital of France is Paris [1]."}
        if system == prompts.VERIFY_PROSE_SYSTEM_PROMPT:
            return {"content": "The capital of France is Paris [1]."}
        if system == prompts.VERIFY_VERDICT_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {"coverage_complete": True, "missing": [], "notes": []}
                )
            }
        if system == prompts.EVIDENCE_LEDGER_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "answer_ready": True,
                        "ledger": [
                            {
                                "requirement": "state the capital",
                                "proposed_claim": "The capital of France is Paris.",
                                "source_ids": [1],
                                "support_level": "supported",
                                "can_use_in_answer": True,
                                "missing": "",
                            }
                        ],
                        "global_missing": [],
                        "next_steps": [],
                        "reason": "The tool output supports the answer.",
                    }
                )
            }
        if system == prompts.EVIDENCE_CHALLENGE_SYSTEM_PROMPT:
            return {
                "content": json.dumps(
                    {
                        "answer_permitted": True,
                        "blocking_gaps": [],
                        "next_steps": [],
                        "reason": "No blocking evidence weakness.",
                    }
                )
            }
        raise AssertionError(f"unexpected system prompt: {system[:80]}")

    async def fake_call_mcp_tool(name, args, session=None):
        assert name == "get_fact"
        return "The capital of France is Paris."

    class FakeStore:
        def best_strategy(self):
            return None

        def record_strategy(self, *a, **k):
            return {}

        def save_skill(self, *a, **k):
            pass

        def add_experience(self, *a, **k):
            pass

    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(memory, "_get_research_store", lambda: FakeStore())

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_fact",
                "description": "Look up a short fact about a topic.",
                "parameters": {
                    "type": "object",
                    "properties": {"topic": {"type": "string"}},
                    "required": ["topic"],
                },
            },
        }
    ]
    graph = agent_module.build_graph("fake-model", tools=tools, profile=GENERIC_PROFILE)
    state = {
        "task": "What is the capital of France?",
        "conversation_context": "",
        "requirements_result": {},
        "plan": [],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "evidence_ledger_result": {},
        "evidence_challenge_result": {},
        "draft_result": {},
        "verification_result": {},
        "evidence_audit": {},
        "final_answer": "",
        "iteration": 0,
        "replan_count": 0,
        "evidence_round": 0,
        "search_memory": agent_module._default_search_memory(),
    }

    final = asyncio.run(graph.ainvoke(state, {"recursion_limit": 200}))

    assert final["final_answer"] == "The capital of France is Paris [1]."
    assert final["verification_result"]["task_complete"] is True
    assert any(s["kind"] == "tool_output" for s in final["sources"])
    assert final["sources"][0]["content"] == "The capital of France is Paris."


def test_evidence_ledger_drops_unindexed_gap_and_recovers_answer_ready():
    """A blocking_gaps/global_missing item with no valid requirement_index cannot be tied
    to a real PROOF_REQUIREMENTS item, so it must not be able to block answer_ready — this
    is the regression test for the 'moscow news' scope-creep bug (invented sub-requirements
    like 'political news for the exact date' blocking readiness even though the only real
    requirement was 'must contain Moscow news')."""
    sources = [
        {
            "title": "Source",
            "url": "https://example.com/a",
            "content": "Moscow news roundup.",
        }
    ]
    completion_criteria = ["The response must contain news related to Moscow."]

    raw = {
        "answer_ready": False,
        "ledger": [
            {
                "claim_id": "moscow-news",
                "requirement_index": 0,
                "support_status": "supported",
                "support_level": "supported",
                "can_use_in_answer": True,
                "source_ids": [1],
                "missing": "",
            }
        ],
        "global_missing": [
            {
                "requirement_index": 5,
                "text": "Political news for the exact date is missing.",
            },
        ],
        "next_steps": [],
        "reason": "Not all categories covered.",
    }

    result = _normalize_evidence_ledger_result(
        raw, sources, completion_criteria, "strict"
    )

    assert result["dropped_gaps"] == ["Political news for the exact date is missing."]
    assert result["global_missing"] == []
    assert result["answer_ready"] is True


def test_evidence_ledger_roundup_mode_answer_ready_from_source_count():
    sources = [
        {"title": "A", "url": "https://example.com/a", "content": "..."},
        {"title": "B", "url": "https://example.com/b", "content": "..."},
        {"title": "C", "url": "https://example.com/c", "content": "..."},
    ]
    completion_criteria = ["The response must contain news related to Moscow."]

    def ledger_row(source_id):
        return {
            "claim_id": f"c{source_id}",
            "requirement_index": 0,
            "support_status": "supported",
            "support_level": "supported",
            "can_use_in_answer": True,
            "source_ids": [source_id],
        }

    raw_three = {
        "answer_ready": False,
        "ledger": [ledger_row(1), ledger_row(2), ledger_row(3)],
        "global_missing": [],
        "next_steps": [],
    }
    result_three = _normalize_evidence_ledger_result(
        raw_three, sources, completion_criteria, "roundup"
    )
    assert result_three["answer_ready"] is True

    raw_two = {
        "answer_ready": False,
        "ledger": [ledger_row(1), ledger_row(2)],
        "global_missing": [],
        "next_steps": [],
    }
    result_two = _normalize_evidence_ledger_result(
        raw_two, sources, completion_criteria, "roundup"
    )
    assert result_two["answer_ready"] is False


def test_evidence_ledger_roundup_does_not_require_one_row_per_requirement():
    sources = [
        {"title": "A", "url": "https://example.com/a", "content": "..."},
        {"title": "B", "url": "https://example.com/b", "content": "..."},
        {"title": "C", "url": "https://example.com/c", "content": "..."},
    ]
    completion_criteria = [
        "The items are current.",
        "The items match the requested subject.",
    ]
    raw = {
        "answer_ready": False,
        "ledger": [
            {
                "claim_id": "current_requested_items",
                "requirement_index": 0,
                "support_status": "supported",
                "support_level": "supported",
                "can_use_in_answer": True,
                "source_ids": [1, 2, 3],
            }
        ],
        "global_missing": [],
        "next_steps": [],
    }

    result = _normalize_evidence_ledger_result(
        raw, sources, completion_criteria, "roundup"
    )

    assert result["admissible_count"] == 3
    assert result["answer_ready"] is True


def test_evidence_ledger_roundup_accepts_several_supported_claims_from_one_source():
    sources = [
        {
            "title": "Requested items",
            "url": "https://example.com/items",
            "content": "...",
        }
    ]
    completion_criteria = [
        "The answer contains several requested items.",
        "The items are current.",
    ]

    def ledger_row(claim_id, requirement_index):
        return {
            "claim_id": claim_id,
            "requirement_index": requirement_index,
            "support_status": "supported",
            "support_level": "supported",
            "can_use_in_answer": True,
            "source_ids": [1],
        }

    raw = {
        "answer_ready": False,
        "ledger": [
            ledger_row("news_1", 0),
            ledger_row("news_2", 0),
            ledger_row("freshness", 1),
        ],
        "global_missing": [],
        "next_steps": [],
    }

    result = _normalize_evidence_ledger_result(
        raw, sources, completion_criteria, "roundup"
    )

    assert result["supported_claim_count"] == 3
    assert result["admissible_count"] == 1
    assert result["answer_ready"] is True


def test_evidence_ledger_dedup_stalled_routes_to_reextract(monkeypatch):
    def fake_chat(
        model,
        messages,
        tools=None,
        system="",
        temperature=0,
        num_predict=32768,
        format_schema=None,
        json_mode=False,
    ):
        return {
            "content": json.dumps(
                {
                    "answer_ready": False,
                    "ledger": [],
                    "global_missing": [],
                    "next_steps": ["web_read: https://example.com/already-read"],
                    "reason": "need more",
                }
            )
        }

    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)

    state = {
        "task": "Find the requested fact",
        "requirements_result": {
            "answer_mode": "strict",
            "completion_criteria": ["Provide the requested fact."],
        },
        "candidate_sources": [
            {"title": "S", "url": "https://example.com/already-read", "content": "text"}
        ],
        "sources": [],
        "completed_steps": [{"step": "web_read: https://example.com/already-read"}],
        "plan": [],
        "evidence_round": 0,
        "iteration": 0,
    }

    update = asyncio.run(
        evidence_ledger_node(state, "fake-model", [], FOOTNOTE_PROFILE)
    )

    assert update["evidence_audit"]["dedup_stalled"] is True
    assert route_after_evidence_ledger(state | update) == "evidence_reextract"


def test_route_after_evidence_reextract_returns_challenge_when_recovered():
    assert (
        route_after_evidence_reextract({"evidence_audit": {"passed": True}})
        == "evidence_challenge"
    )


def test_route_after_evidence_reextract_returns_assimilate_when_not_recovered():
    assert (
        route_after_evidence_reextract({"evidence_audit": {"passed": False}})
        == "assimilate"
    )


def test_route_after_evidence_reextract_returns_partial_answer_when_supported_sources_remain():
    state = {
        "evidence_audit": {"passed": False},
        "sources": [
            {"url": "https://example.com/supported", "content": "Supported fact."}
        ],
    }

    assert route_after_evidence_reextract(state) == "answer"


def test_graph_can_route_failed_reextract_with_sources_to_partial_answer(monkeypatch):
    async def fake_requirements(state, model, tools, profile):
        return {
            "requirements_result": {
                "completion_criteria": ["Return all requested items."]
            },
            "iteration": state["iteration"] + 1,
        }

    async def fake_plan(state, model, tools, profile):
        return {"plan": ["example_tool: request"], "iteration": state["iteration"] + 1}

    async def fake_execute(state, model, tools, profile, mcp_session=None):
        return {
            "plan": [],
            "completed_steps": [
                {
                    "step": "example_tool: request",
                    "result": "supported",
                    "tools_used": [],
                }
            ],
            "candidate_sources": [
                {"title": "S", "url": "https://example.com/s", "content": "Supported."}
            ],
            "sources": [
                {"title": "S", "url": "https://example.com/s", "content": "Supported."}
            ],
            "iteration": state["iteration"] + 1,
        }

    async def fake_ledger(state, model, tools, profile):
        return {
            "evidence_audit": {"passed": False, "dedup_stalled": True},
            "iteration": state["iteration"] + 1,
        }

    async def fake_reextract(state, model, tools, profile):
        return {
            "evidence_audit": {
                "passed": False,
                "gaps": ["One requested item is missing."],
            },
            "iteration": state["iteration"] + 1,
        }

    async def fake_answer(state, model, tools, profile):
        return {
            "draft_result": {
                "answer": "Partial supported answer.",
                "salvaged_prose": True,
            },
            "iteration": state["iteration"] + 1,
        }

    async def fake_verify(state, model, tools, profile):
        return {
            "final_answer": "Partial supported answer.",
            "verification_result": {
                "task_complete": False,
                "gaps": ["One requested item is missing."],
            },
            "iteration": state["iteration"] + 1,
        }

    async def fake_assimilate(state, model, tools):
        return {"iteration": state["iteration"] + 1}

    monkeypatch.setattr(nodes_module, "requirements_node", fake_requirements)
    monkeypatch.setattr(nodes_module, "plan_node", fake_plan)
    monkeypatch.setattr(nodes_module, "execute_node", fake_execute)
    monkeypatch.setattr(nodes_module, "evidence_ledger_node", fake_ledger)
    monkeypatch.setattr(nodes_module, "evidence_reextract_node", fake_reextract)
    monkeypatch.setattr(nodes_module, "answer_node", fake_answer)
    monkeypatch.setattr(nodes_module, "verify_node", fake_verify)
    monkeypatch.setattr(nodes_module, "assimilate_node", fake_assimilate)

    graph = nodes_module.build_graph("fake-model", tools=[])
    state = {
        "task": "Return every requested item",
        "conversation_context": "",
        "requirements_result": {},
        "plan": [],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "evidence_ledger_result": {},
        "evidence_challenge_result": {},
        "draft_result": {},
        "verification_result": {},
        "evidence_audit": {},
        "final_answer": "",
        "iteration": 0,
        "replan_count": 0,
        "evidence_round": 0,
        "search_memory": agent_module._default_search_memory(),
        "answer_mode": "strict",
        "stagnant_rounds": 0,
        "last_supported_claim_count": 0,
    }

    final = asyncio.run(graph.ainvoke(state))

    assert final["final_answer"] == "Partial supported answer."


def test_sources_from_tool_result_passes_through_listing_metadata():
    payload = {
        "url": "https://example.com/news",
        "title": "Latest News",
        "text": "Listing page text.",
        "is_listing": True,
        "links": [
            {"text": "Headline one", "url": "https://example.com/a1"},
            {"text": "Headline two", "url": "https://example.com/a2"},
        ],
    }
    sources = agent_module._sources_from_tool_result("web_read", json.dumps(payload))

    assert sources[0]["is_listing_page"] is True
    assert sources[0]["candidate_links"] == [
        {"text": "Headline one", "url": "https://example.com/a1"},
        {"text": "Headline two", "url": "https://example.com/a2"},
    ]


def test_execute_node_auto_enqueues_drilldown_steps_for_listing_page(monkeypatch):
    async def fake_call_mcp_tool(name, args, session=None):
        assert name == "web_read"
        return json.dumps(
            {
                "url": "https://example.com/news",
                "title": "Latest News",
                "text": "Listing page text with several headlines.",
                "is_listing": True,
                "links": [
                    {
                        "text": "City announces new metro line",
                        "url": "https://example.com/a1",
                    },
                    {
                        "text": "Mayor holds press conference",
                        "url": "https://example.com/a2",
                    },
                    {"text": "Weather warning issued", "url": "https://example.com/a3"},
                ],
            }
        )

    def fake_ollama_chat_json(*args, **kwargs):
        return json.dumps({"decision": "DONE", "reason": "listing page read"})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(llm, "_ollama_chat_schema", fake_ollama_chat_json)
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    state = {
        "task": "Find Moscow news",
        "requirements_result": {},
        "plan": ["web_read: https://example.com/news"],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "search_memory": {
            **agent_module._default_search_memory(),
            "discovered_urls": ["https://example.com/news"],
        },
        "iteration": 0,
    }

    update = asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE))

    assert update["plan"] == [
        "web_read: https://example.com/a1",
        "web_read: https://example.com/a2",
        "web_read: https://example.com/a3",
    ]


def test_invented_urls_are_dropped_and_discovered_ones_kept():
    from llmflow_search.nodes import _drop_undiscovered_url_steps

    known = {"https://www.lamoncloa.gob.es/real-page"}
    kept, invented = _drop_undiscovered_url_steps(
        [
            "web_read: https://www.lamoncloa.gob.es/real-page",
            "web_read: https://es.wikipedia.org/wiki/Estrategia_Nacional_de_Inteligencia_Artificial",
            "web_search: España ENIA estrategia",
        ],
        known,
    )
    assert kept == [
        "web_read: https://www.lamoncloa.gob.es/real-page",
        "web_search: España ENIA estrategia",
    ]
    assert invented == [
        "web_read: https://es.wikipedia.org/wiki/Estrategia_Nacional_de_Inteligencia_Artificial"
    ]


def test_search_results_are_recorded_as_discovered_not_read():
    import json as _json

    from llmflow_search.search_memory import _merge_search_memory, _update_search_memory

    memory = _merge_search_memory(None)
    payload = _json.dumps(
        {
            "sources": [
                {"url": "https://example.gov/a", "title": "Programa nacional"},
                {"url": "https://example.gov/b", "title": "Segundo"},
            ],
            "count": 2,
        }
    )
    memory = _update_search_memory(memory, "web_search", {"query": "q"}, payload)

    assert memory["discovered_urls"] == [
        "https://example.gov/a",
        "https://example.gov/b",
    ]
    assert memory["discovered_titles"]["https://example.gov/a"] == "Programa nacional"
    # Finding a URL is not reading it: read_urls must stay empty until a fetch runs.
    assert memory["read_urls"] == []


def test_web_search_results_are_recorded_under_their_real_key():
    import json as _json

    from llmflow_search.search_memory import _merge_search_memory, _update_search_memory

    memory = _merge_search_memory(None)
    # footnote's web_search returns "results", not "sources".
    payload = _json.dumps(
        {
            "query": "q",
            "count": 2,
            "results": [
                {"url": "https://coinstats.app/a", "title": "Bitcoin price"},
                {"url": "https://example.gov/b", "title": "Second"},
            ],
        }
    )
    memory = _update_search_memory(memory, "web_search", {"query": "q"}, payload)

    assert memory["discovered_urls"] == [
        "https://coinstats.app/a",
        "https://example.gov/b",
    ]
    assert memory["discovered_titles"]["https://coinstats.app/a"] == "Bitcoin price"


def test_search_snippets_and_ranks_are_recorded_for_the_read_decision():
    """Dropping the snippet left the post-batch model choosing what to read from a bare
    address list, which is how a run reads three junk pages out of a hundred found."""
    import json as _json

    from llmflow_search.search_memory import _merge_search_memory, _update_search_memory

    memory = _merge_search_memory(None)
    payload = _json.dumps(
        {
            "query": "q",
            "results": [
                {
                    "url": "https://example.gov/a",
                    "title": "First",
                    "snippet": "The  university  was founded\nin 1996.",
                },
                {"url": "https://example.gov/b", "title": "Second"},
            ],
        }
    )
    memory = _update_search_memory(memory, "web_search", {"query": "q"}, payload)

    assert (
        memory["discovered_snippets"]["https://example.gov/a"]
        == "The university was founded in 1996."
    )
    assert memory["discovered_ranks"] == {
        "https://example.gov/a": 0,
        "https://example.gov/b": 1,
    }

    # A URL that later comes back as a top hit is promoted to its best rank.
    payload = _json.dumps(
        {"query": "q2", "results": [{"url": "https://example.gov/b", "title": "Second"}]}
    )
    memory = _update_search_memory(memory, "web_search", {"query": "q2"}, payload)
    assert memory["discovered_ranks"]["https://example.gov/b"] == 0


def test_execute_node_auto_reads_top_ranked_pages_in_a_later_round(monkeypatch):
    """The rule used to fire only while nothing had been read at all, so a run that
    fetched one page early spent every later round on snippets it never opened."""

    async def fake_call_mcp_tool(name, args, session=None):
        assert name == "web_search"
        return json.dumps(
            {
                "query": args.get("query"),
                "results": [
                    {
                        "url": "https://example.gov/hit",
                        "title": "Hit",
                        "snippet": "Founded in 1996.",
                    },
                    {"url": "https://example.gov/second", "title": "Second"},
                ],
            }
        )

    asked = []

    def fake_ollama_chat_json(model, messages, system=None, **kwargs):
        asked.append(system)
        return json.dumps(
            {
                "decision": "NEXT",
                "reason": "keep searching",
                "next_steps": ["web_search: another angle"],
            }
        )

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(llm, "_ollama_chat_schema", fake_ollama_chat_json)
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    state = {
        "task": "Name the institution",
        "requirements_result": {},
        "plan": ["web_search: institution founded 1996"],
        "completed_steps": [],
        "scratchpad": "",
        # A page was already fetched this run: the old guard would stay silent from here on.
        "candidate_sources": [{"url": "https://example.gov/old", "content": "text"}],
        "admissible_sources": [],
        "sources": [],
        "search_memory": {
            **agent_module._default_search_memory(),
            # Arrived first, but from a weak opening query and with no rank of its own.
            "discovered_urls": ["https://example.gov/stale"],
            "read_urls": ["https://example.gov/old"],
        },
        "iteration": 0,
    }

    update = asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE))

    assert update["plan"][:2] == [
        "web_read: https://example.gov/hit",
        "web_read: https://example.gov/second",
    ]
    assert "web_read: https://example.gov/stale" in update["plan"]
    # The reads are settled by the state, so the controller is not asked for an opinion
    # whose only effect would be to queue more searching behind pages nobody has opened.
    assert FOOTNOTE_PROFILE.post_batch not in asked


def test_a_url_step_with_nothing_seen_anywhere_is_an_invention():
    """Leniency here was the hole the planner walked through: its steps are written
    before any search runs, so "nothing discovered yet" described every initial plan."""
    from llmflow_search.nodes import _drop_undiscovered_url_steps

    steps = ["web_read: https://example.gov/a", "web_search: q"]
    kept, invented = _drop_undiscovered_url_steps(steps, set())

    assert kept == ["web_search: q"]  # a search needs no prior address
    assert invented == ["web_read: https://example.gov/a"]


def test_an_address_the_task_supplied_is_never_called_an_invention():
    from llmflow_search.nodes import _drop_undiscovered_url_steps
    from llmflow_search.sources import _urls_in_text

    task = "Summarise https://example.gov/report for me"
    kept, invented = _drop_undiscovered_url_steps(
        ["web_read: https://example.gov/report"], _urls_in_text(task)
    )

    assert kept == ["web_read: https://example.gov/report"]
    assert invented == []


def test_an_address_printed_inside_a_fetched_page_counts_as_seen():
    """A citation the agent read with its own eyes is not an invention, even though no
    search returned it."""
    from llmflow_search.nodes import _drop_undiscovered_url_steps
    from llmflow_search.sources import _urls_in_sources

    sources = [
        {
            "url": "https://example.gov/index",
            "content": "The figures are published at https://example.gov/data/2026.csv each quarter.",
        }
    ]
    kept, invented = _drop_undiscovered_url_steps(
        ["web_read: https://example.gov/data/2026.csv"], _urls_in_sources(sources)
    )

    assert kept == ["web_read: https://example.gov/data/2026.csv"]
    assert invented == []


def test_links_found_on_a_fetched_page_count_as_discovered():
    import json as _json

    from llmflow_search.search_memory import _merge_search_memory, _update_search_memory

    memory = _merge_search_memory(None)
    payload = _json.dumps(
        {
            "url": "https://example.gov/index",
            "text": "content",
            "links": [{"url": "https://example.gov/report.pdf", "text": "Report"}],
        }
    )
    memory = _update_search_memory(
        memory, "web_read", {"url": "https://example.gov/index"}, payload
    )

    assert "https://example.gov/report.pdf" in memory["discovered_urls"]
    assert memory["read_urls"] == ["https://example.gov/index"]


def _load_browsecomp_stub(monkeypatch, tmp_path, documents):
    """Import the benchmark's stdio MCP server with a corpus already loaded."""
    import importlib.util
    import json as _json
    from pathlib import Path

    corpus = tmp_path / "corpus.json"
    corpus.write_text(_json.dumps(documents), encoding="utf-8")
    monkeypatch.setenv("BROWSECOMP_DOCS_JSON", str(corpus))

    script = Path(__file__).resolve().parent.parent / "scripts" / "browsecomp_mcp_server.py"
    spec = importlib.util.spec_from_file_location("browsecomp_mcp_server_under_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_corpus()
    return module


def test_benchmark_corpus_is_reachable_at_the_address_the_agent_records(monkeypatch, tmp_path):
    """The corpus held "…/ls054609505/" and the agent asked for "…/ls054609505": every
    read of that page returned "Document not found" while the document sat in the corpus,
    so part of the benchmark could not be answered no matter what the agent chose."""
    import json as _json

    from llmflow_search.search_memory import _merge_search_memory, _update_search_memory

    stub = _load_browsecomp_stub(
        monkeypatch,
        tmp_path,
        [{"docid": "d1", "url": "https://www.imdb.com/list/ls054609505/", "title": "List", "text": "Kirk Douglas appears in the list of actors born in 1916."}],
    )

    results = _json.loads(stub.web_search("actors born 1916", count=5))
    memory = _update_search_memory(
        _merge_search_memory(None), "web_search", {"query": "actors born 1916"}, _json.dumps(results)
    )
    recorded = memory["discovered_urls"][0]

    assert recorded == "https://www.imdb.com/list/ls054609505"
    page = _json.loads(stub.web_read(recorded))
    assert "error" not in page
    assert page["text"].startswith("Kirk Douglas")


def test_a_server_declares_its_own_pacing(monkeypatch):
    """Which backend a tool talks to is the server's knowledge. The name map guessed
    "scraper" for anything called web_search, so a local BM25 stub slept twelve seconds
    before reading a file off disk."""
    from llmflow_search.config import group_batch_cap

    catalog = [
        {"function": {"name": "web_search", "description": "BM25 over a local corpus. [throttle: local]"}},
        {"function": {"name": "web_read", "description": "Read a local document."}},
    ]
    monkeypatch.setattr(mcp_client, "_declared_throttle_groups", {})
    mcp_client.configure_tool_pacing(catalog)

    assert mcp_client._throttle_group("web_search") == "local"
    # A family that never pauses has no pauses to spread across rounds.
    assert group_batch_cap("local") > group_batch_cap("scraper")
    # An undeclared tool keeps the footnote-mcp default.
    mcp_client.configure_tool_pacing([{"function": {"name": "web_search", "description": "Search the web."}}])
    assert mcp_client._throttle_group("web_search") == "scraper"


def test_a_declared_unmetered_tool_never_waits(monkeypatch):
    slept: list[float] = []

    async def fake_sleep(seconds):
        slept.append(seconds)

    monkeypatch.setattr(mcp_client.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(mcp_client, "_declared_throttle_groups", {})
    mcp_client.configure_tool_pacing(
        [{"function": {"name": "web_search", "description": "Local search. [throttle: none]"}}]
    )

    assert mcp_client._throttle_group("web_search") is None
    asyncio.run(mcp_client._wait_for_search_request_slot("web_search"))
    asyncio.run(mcp_client._wait_for_search_request_slot("web_search"))
    assert slept == []


def _load_analyze_run():
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "analyze_run.py"
    spec = importlib.util.spec_from_file_location("analyze_run_under_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_run_leaves_a_trace_the_three_metrics_can_be_read_from(monkeypatch, tmp_path):
    """Retriever quality, page-selection quality and synthesis quality used to be
    indistinguishable: a run left one score behind, and everything else was recovered by
    running regular expressions over prose meant for a human."""
    from llmflow_search import trace as trace_module

    async def fake_call_mcp_tool(name, args, session=None):
        if name == "web_search":
            return json.dumps(
                {
                    "results": [
                        {"url": "https://example.gov/gold", "title": "Gold", "snippet": "Founded 1886."},
                        {"url": "https://example.gov/noise", "title": "Noise", "snippet": "Unrelated."},
                    ]
                }
            )
        return json.dumps({"url": args.get("url"), "title": "t", "text": "Founded in 1886."})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "NEXT", "reason": "read it", "next_steps": []}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    trace_path = tmp_path / "run.jsonl"
    trace_module.start_run(trace_path)
    try:
        trace_module.begin_query("q1", gold_urls=["https://example.gov/gold"], corpus_size=42)
        state = {
            "task": "When was it founded?",
            "requirements_result": {},
            "plan": ["web_search: founded"],
            "completed_steps": [],
            "scratchpad": "",
            "candidate_sources": [],
            "admissible_sources": [],
            "sources": [],
            "search_memory": agent_module._default_search_memory(),
            "iteration": 0,
        }
        update = asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE))
        # Reading the pages the round queued is what the next execute call does; run it
        # so the trace holds a real web_read, not just the intention to make one.
        state = {**state, **update}
        asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE))
        trace_module.emit("query_end", is_exact_match=True, hit_gold_sources=True, score=1.0, elapsed_seconds=3.5)
    finally:
        trace_module.close_run()

    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    kinds = {event["event"] for event in events}
    assert {"search_results", "read_decision", "post_batch_decision", "tool_call", "llm_call"} <= kinds
    assert all(event["run_id"] and event["query_id"] == "q1" for event in events if event["event"] != "query_start" or True)

    analyze = _load_analyze_run()
    metrics = analyze.analyze_query("q1", events)

    assert metrics["best_gold_rank"] == 0  # retriever put the gold document first
    assert metrics["gold_read"] == 1 and metrics["gold_read_rate"] == 1.0
    assert metrics["read_precision_gold"] == 0.5  # one of the two pages opened was gold
    assert metrics["is_exact_match"] is True  # synthesis
    assert metrics["llm_calls_by_role"]["post_batch"] >= 1
    assert metrics["seconds_by_event"]["tool_call"] >= 0


def test_the_transition_table_reads_from_one_module():
    """Six guards used to repair the controller's answer downstream of it, each added
    after a specific failure. They are the same six rules; they are now the decision."""
    from llmflow_search.policy import RoundView, decide

    # A finished run needs a fetched source. Snippets are not evidence, and accepting
    # DONE here is what let a run report success and leave the ledger with nothing.
    view = RoundView(remaining=("web_search: b",), has_sources=False)
    assert decide(view, "DONE").steps == ("web_search: b",)
    assert decide(RoundView(remaining=("web_search: b",), has_sources=True), "DONE").finishes

    # CONTINUE with nothing left to continue is a no-op that ends the round having
    # fetched nothing.
    assert decide(RoundView(has_sources=True), "CONTINUE").finishes

    # Every condition settled by a fetched source: more searching cannot improve on it.
    settled = RoundView(remaining=("web_search: c",), has_sources=True, total_conditions=3)
    assert decide(settled, "NEXT", ["web_search: d"]).finishes

    # An address nobody ever returned, and a step already executed, are both refused —
    # and the refusal is reported rather than silently applied.
    view = RoundView(
        known_urls=frozenset({"https://seen.example/a"}),
        completed_identities=frozenset({"web_search:old"}),
        has_sources=True,
        remaining=("web_search: keep",),
    )
    action = decide(
        view,
        "NEXT",
        ["web_read: https://invented.example/x", "web_search: old", "web_search: fine"],
    )
    assert action.steps == ("web_search: fine", "web_search: keep")
    assert dict(action.rejected) == {
        "web_read: https://invented.example/x": "undiscovered_url",
        "web_search: old": "already_executed",
    }


def test_forced_reads_compose_with_the_controller_rather_than_replacing_it():
    """A decision to keep searching and a page that must be opened are not in conflict.
    Reading comes first; the controller's admissible steps queue behind it."""
    from llmflow_search.policy import RoundView, decide, forced_reads

    view = RoundView(
        remaining=("web_search: planned",),
        ranked_unread=("https://example.org/top", "https://example.org/next"),
        known_urls=frozenset({"https://example.org/top", "https://example.org/next"}),
        tool_just_run="web_search",
        has_sources=True,
    )
    forced, by = forced_reads(view)
    assert by == "auto_rank" and forced[0] == "web_read: https://example.org/top"

    action = decide(view, "NEXT", ["web_search: another"], forced, by)
    assert action.steps[: len(forced)] == tuple(forced)
    assert action.steps[-2:] == ("web_search: another", "web_search: planned")


def test_a_listing_page_is_opened_even_when_the_controller_says_done():
    """Finishing on the index page is finishing without the content."""
    from llmflow_search.policy import RoundView, decide, forced_reads

    view = RoundView(drilldown=("https://example.org/article",), has_sources=True)
    forced, by = forced_reads(view)

    action = decide(view, "DONE", [], forced, by)
    assert action.steps == ("web_read: https://example.org/article",)
    assert action.source == "listing_drilldown"


def test_the_controller_is_still_asked_on_every_batch_when_the_flag_is_off(monkeypatch):
    from llmflow_search import policy as policy_module

    view = policy_module.RoundView(
        ranked_unread=("https://example.org/a",),
        tool_just_run="web_search",
    )
    forced, _ = policy_module.forced_reads(view)
    assert forced

    monkeypatch.setattr(policy_module, "SKIP_CONTROLLER_WHEN_READS_FORCED", False)
    assert policy_module.should_ask_controller(view, forced) is True
    monkeypatch.setattr(policy_module, "SKIP_CONTROLLER_WHEN_READS_FORCED", True)
    assert policy_module.should_ask_controller(view, forced) is False


def test_the_trace_records_the_controller_answer_and_the_action_taken(monkeypatch, tmp_path):
    """Knowing how often the state overrules the controller is the only way to tell a
    load-bearing rule from dead weight."""
    from llmflow_search import trace as trace_module

    async def fake_call_mcp_tool(name, args, session=None):
        return json.dumps({"url": args.get("url"), "error": "not found", "text": ""})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "DONE", "reason": "snippets look right"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    memory = agent_module._default_search_memory()
    memory["discovered_urls"] = ["https://example.gov/a"]

    trace_path = tmp_path / "run.jsonl"
    trace_module.start_run(trace_path)
    try:
        trace_module.begin_query("q2", gold_urls=[])
        state = {
            "task": "q",
            "requirements_result": {},
            "plan": ["web_read: https://example.gov/a", "web_search: fallback"],
            "completed_steps": [],
            "scratchpad": "",
            "candidate_sources": [],
            "admissible_sources": [],
            "sources": [],
            "search_memory": memory,
            "iteration": 0,
        }
        update = asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE))
    finally:
        trace_module.close_run()

    # The read produced no source, so DONE cannot stand and the plan survives.
    assert update["plan"] == ["web_search: fallback"]

    decisions = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "post_batch_decision"
    ]
    assert decisions[0]["asked_model"] is True
    assert decisions[0]["model_decision"] == "DONE"
    assert decisions[0]["decision"] == "RUN"


def test_emitting_without_an_open_trace_is_a_no_op():
    from llmflow_search import trace as trace_module

    trace_module.close_run()
    trace_module.emit("anything", value=1)  # must not raise
    assert trace_module.active() is None


def test_context_truncation_keeps_the_part_that_answers_the_question():
    """A long page was cut to its first N characters — on a long page, the masthead. The
    sentence the question turns on sat past the cut, and the model was then asked why it
    could not find it."""
    from llmflow_search.sources import _format_sources_for_llm

    boilerplate = "Navigation home about contact subscribe newsletter archive. " * 600
    answer = "The observatory was commissioned in 1886 by the Royal Society."
    page = boilerplate + answer + " " + boilerplate

    sources = [{"title": "Page", "url": "https://example.org/a", "content": page, "kind": "page"}]
    relevant, _ = _format_sources_for_llm(sources, "When was the observatory commissioned?")
    positional, _ = _format_sources_for_llm(sources)

    assert "commissioned in 1886" in relevant
    assert "commissioned in 1886" not in positional  # what the old behaviour showed
    assert len(relevant) < len(page)


def test_a_short_page_is_shown_whole_whatever_the_query():
    from llmflow_search.sources import _format_sources_for_llm

    sources = [{"title": "T", "url": "https://example.org/a", "content": "Founded 1886.", "kind": "page"}]
    text, ids = _format_sources_for_llm(sources, "unrelated question about turnips")

    assert "Founded 1886." in text and ids == {1}


def test_the_scoring_implementation_can_be_replaced_without_touching_the_caller():
    """The seam is the deliverable: a dense scorer has to drop in where BM25 is, and
    nodes.py must not learn that it happened."""
    from llmflow_search.passages import Bm25PassageScorer, PassageStore, get_scorer, set_scorer

    memory = {
        "discovered_urls": ["https://example.org/a", "https://example.org/b"],
        "discovered_snippets": {
            "https://example.org/a": "Unrelated municipal budget minutes.",
            "https://example.org/b": "The observatory was commissioned in 1886.",
        },
        "discovered_titles": {},
        "discovered_ranks": {"https://example.org/a": 0, "https://example.org/b": 1},
    }
    catalog = ["https://example.org/a", "https://example.org/b"]

    store = PassageStore().add_search_snippets(memory)
    assert store.rank_urls("observatory commissioned 1886", catalog)[0] == "https://example.org/b"

    class ReverseScorer:
        """Stands in for a dense scorer: same one method, different opinion."""

        def score(self, query, passages):
            return [1.0 if "municipal" in p.text else 0.0 for p in passages]

    original = get_scorer()
    try:
        set_scorer(ReverseScorer())
        # Same store construction, same call — only the installed implementation changed.
        flipped = PassageStore().add_search_snippets(memory)
        assert flipped.rank_urls("observatory commissioned 1886", catalog)[0] == "https://example.org/a"
    finally:
        set_scorer(original)
    assert isinstance(get_scorer(), Bm25PassageScorer)


def test_read_candidates_are_chosen_through_the_passage_layer():
    """Which comparison orders them is READ_RANKING's business; that the passage store
    is what does the ordering is not."""
    from llmflow_search.nodes import _rank_read_candidates

    memory = {
        "discovered_urls": ["https://example.org/old", "https://example.org/new"],
        "discovered_snippets": {},
        "discovered_titles": {},
        # Same rank from two different queries: the more recent query is the one the run
        # refined toward, so its hit goes first.
        "discovered_ranks": {"https://example.org/old": 0, "https://example.org/new": 0},
    }
    catalog = ["https://example.org/old", "https://example.org/new"]

    assert _rank_read_candidates("q", catalog, memory) == [
        "https://example.org/new",
        "https://example.org/old",
    ]


def test_a_top_ranked_result_with_no_snippet_still_gets_read():
    """Not every search result carries a description; dropping the ones that do not
    would silently hide the best hit."""
    from llmflow_search.nodes import _rank_read_candidates

    memory = {
        "discovered_urls": ["https://example.org/described", "https://example.org/bare"],
        "discovered_snippets": {"https://example.org/described": "Some prose."},
        "discovered_titles": {},
        "discovered_ranks": {"https://example.org/described": 4, "https://example.org/bare": 0},
    }
    catalog = ["https://example.org/described", "https://example.org/bare"]

    assert _rank_read_candidates("q", catalog, memory)[0] == "https://example.org/bare"


def test_passages_carry_where_they_came_from():
    from llmflow_search.passages import PassageStore

    store = PassageStore().add_sources(
        [{"url": "https://example.org/a", "title": "A", "content": "x" * 2500}]
    )
    passages = store.ranked("x", origin="page")

    assert len(passages) > 1  # a long page is more than one passage
    offsets = sorted(p.offset for _, p in passages)
    assert offsets[0] == 0 and offsets[1] > 0
    assert all(p.url == "https://example.org/a" and p.origin == "page" for _, p in passages)


def _ledger(rows, **extra):
    return {"ledger": rows, "global_missing": [], "next_steps": [], "reason": "", **extra}


def test_the_registry_says_which_conditions_are_closed_and_by_what():
    """Nothing in the run held the idea of a condition with a status, so a later round
    could go looking again for something an earlier round had already established."""
    from llmflow_search.constraints import build_registry, open_constraints

    criteria = [
        "The person was born in 1886.",
        "The trip ran from April to November.",
        "The person was mistaken for a shaman.",
    ]
    sources = [
        {"url": "https://example.org/bio/", "content": "..."},
        {"url": "https://example.org/diary", "content": "..."},
    ]
    registry = build_registry(
        criteria,
        _ledger(
            [
                {
                    "requirement_index": 0,
                    "support_status": "supported",
                    "can_use_in_answer": True,
                    "source_ids": [1],
                    "proposed_claim": "Born 4 March 1886 in Kazan.",
                },
                {
                    "requirement_index": 1,
                    "support_status": "partial",
                    "can_use_in_answer": False,
                    "source_ids": [2],
                    "proposed_claim": "Departed in April; return date not stated.",
                },
            ]
        ),
        sources,
    )

    assert [item["status"] for item in registry] == ["satisfied", "partial", "open"]
    # The trailing slash must not make the same page look like two.
    assert registry[0]["source_urls"] == ["https://example.org/bio"]
    assert registry[0]["quote"].startswith("Born 4 March 1886")
    assert [item["index"] for item in open_constraints(registry)] == [1, 2]


def test_a_refuted_condition_is_settled_not_reopened():
    """A condition proven false is answered. Leaving it "open" sends the run searching
    for something it has already decided."""
    from llmflow_search.constraints import all_settled, build_registry

    registry = build_registry(
        ["The building was completed in 1901."],
        _ledger([{ "requirement_index": 0, "support_status": "rejected", "can_use_in_answer": False, "source_ids": [], "proposed_claim": "Completed 1912, not 1901."}]),
        [],
    )

    assert registry[0]["status"] == "refuted"
    assert all_settled(registry) is True


def test_the_post_batch_controller_is_shown_the_whole_registry(monkeypatch):
    """It used to see scratchpad[-2500:] and nothing else, so run memory was lost exactly
    in the middle — the point at which a long question needs it."""
    seen = {}

    async def fake_call_mcp_tool(name, args, session=None):
        return json.dumps({"url": args.get("url"), "title": "S", "text": "body"})

    def capture(model, messages, system=None, format_schema=None, **kwargs):
        if system == FOOTNOTE_PROFILE.post_batch:
            seen["prompt"] = messages[0]["content"]
        return json.dumps({"decision": "NEXT", "reason": "x", "next_steps": []})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(llm, "_ollama_chat_schema", capture)
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    state = {
        "task": "Who was it?",
        "requirements_result": {"completion_criteria": ["Born in 1886.", "Mistaken for a shaman."]},
        # A read batch: nothing forces the next reads, so the controller is consulted.
        "plan": ["web_read: https://example.org/seen"],
        "completed_steps": [],
        "scratchpad": "z" * 40000,  # the tail alone would crowd everything else out
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "constraint_registry": [
            {"index": 0, "text": "Born in 1886.", "status": "satisfied", "source_urls": ["https://example.org/bio"], "quote": "Born 1886."},
            {"index": 1, "text": "Mistaken for a shaman.", "status": "open", "source_urls": [], "quote": ""},
        ],
        "search_memory": {
            **agent_module._default_search_memory(),
            "discovered_urls": ["https://example.org/seen"],
        },
        "iteration": 0,
    }
    asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE))

    prompt = seen["prompt"]
    assert "CONDITION REGISTRY" in prompt
    assert "[OK  ] 0. Born in 1886." in prompt
    assert "[??  ] 1. Mistaken for a shaman." in prompt
    assert "https://example.org/bio" in prompt


def test_the_registry_is_written_to_the_trace_on_every_ledger_pass(monkeypatch, tmp_path):
    from llmflow_search import trace as trace_module

    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps(
            {
                "ledger": [
                    {
                        "claim_id": "c1",
                        "requirement_index": 0,
                        "support_status": "supported",
                        "support_level": "supported",
                        "can_use_in_answer": True,
                        "source_ids": [1],
                        "proposed_claim": "Born 1886.",
                    }
                ],
                "global_missing": [],
                "next_steps": [],
                "reason": "ok",
                "answer_ready": True,
            }
        ),
    )

    state = {
        "task": "q",
        "requirements_result": {"completion_criteria": ["Born in 1886."], "answer_mode": "strict"},
        "candidate_sources": [{"url": "https://example.org/bio", "content": "Born 1886.", "kind": "page"}],
        "sources": [],
        "completed_steps": [],
        "plan": [],
        "iteration": 0,
        "evidence_round": 0,
    }

    trace_path = tmp_path / "run.jsonl"
    trace_module.start_run(trace_path)
    try:
        trace_module.begin_query("q1")
        update = asyncio.run(evidence_ledger_node(state, "main", [], FOOTNOTE_PROFILE))
    finally:
        trace_module.close_run()

    assert update["constraint_registry"][0]["status"] == "satisfied"
    events = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    registry_events = [e for e in events if e["event"] == "constraint_registry"]
    assert registry_events[-1]["constraints"][0]["source_urls"] == ["https://example.org/bio"]


def test_per_step_model_calls_in_one_batch_actually_overlap(monkeypatch):
    """Five pages were fetched in parallel and then diagnosed one after another: the
    Ollama client is synchronous, so calling it straight from a coroutine stalled every
    other step in the batch. The fetches were concurrent; the model calls never were."""
    import time

    concurrent = {"now": 0, "peak": 0}

    async def fake_call_mcp_tool(name, args, session=None):
        return json.dumps({"url": args.get("url"), "title": "t", "text": "body"})

    def slow_observation(model, messages, system=None, **kwargs):
        if system != OBSERVATION_SYSTEM_PROMPT:
            return json.dumps({"decision": "CONTINUE", "reason": "x"})
        concurrent["now"] += 1
        concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
        time.sleep(0.05)
        concurrent["now"] -= 1
        return json.dumps({"useful": True, "summary": "read", "reason": "ok"})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(llm, "_ollama_chat_schema", slow_observation)
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    memory = agent_module._default_search_memory()
    memory["discovered_urls"] = [f"https://example.org/{i}" for i in range(3)]
    state = {
        "task": "q",
        "requirements_result": {},
        "plan": [f"web_read: https://example.org/{i}" for i in range(3)],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "search_memory": memory,
        "iteration": 0,
    }
    asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE))

    assert concurrent["peak"] == 3


def test_cheap_roles_can_run_on_a_second_model(monkeypatch):
    """Latency is the number of decisions times the latency of each, and classifying a
    tool result is not the same kind of decision as judging evidence."""
    from llmflow_search import llm as llm_module

    monkeypatch.setattr(llm_module, "FAST_MODEL", "")
    assert llm_module.model_for_role("main:cloud", "observation") == "main:cloud"

    monkeypatch.setattr(llm_module, "FAST_MODEL", "small:local")
    monkeypatch.setattr(llm_module, "FAST_MODEL_ROLES", frozenset({"observation"}))
    assert llm_module.model_for_role("main:cloud", "observation") == "small:local"
    # Everything that reads evidence or writes an answer stays on the main model.
    for role in ("evidence_ledger", "evidence_challenge", "post_batch", "answer", "verify"):
        assert llm_module.model_for_role("main:cloud", role) == "main:cloud"
