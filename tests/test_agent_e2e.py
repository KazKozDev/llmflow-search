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
)


def test_agent_graph_completes_with_fake_model_and_fake_tool(monkeypatch):
    def fake_chat(model, messages, tools=None, system="", temperature=0.3, json_mode=False, num_predict=32768, format_schema=None):
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
            return {"content": json.dumps({"coverage_complete": True, "missing": [], "notes": []})}
        if system == agent_module.EVIDENCE_LEDGER_SYSTEM_PROMPT:
            return {"content": json.dumps({
                "answer_ready": True,
                "ledger": [{
                    "requirement": "rate value",
                    "proposed_claim": "The sourced EUR/RUB rate is 90.1.",
                    "source_ids": [1],
                    "support_level": "supported",
                    "can_use_in_answer": True,
                    "missing": "",
                }],
                "global_missing": [],
                "next_steps": [],
                "reason": "The fetched source supports the required rate.",
            })}
        if system == agent_module.EVIDENCE_CHALLENGE_SYSTEM_PROMPT:
            return {"content": json.dumps({
                "answer_permitted": True,
                "blocking_gaps": [],
                "next_steps": [],
                "reason": "No blocking evidence weakness.",
            })}
        # execute_node's post-batch controller decides DONE/CONTINUE/NEXT. Its prompt is
        # a local string, so match on stable wording. The web_read source is fetched, so
        # DONE is valid and ends the loop.
        if "After executing a batch of steps" in system:
            return {"content": json.dumps({"decision": "DONE", "reason": "rate is covered"})}
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
        "task": "Find the test rate",
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

    def fake_chat(model, messages, tools=None, system="", temperature=0.3, json_mode=False, num_predict=32768, format_schema=None):
        nonlocal ledger_calls, challenge_calls
        if system == agent_module.REQUIREMENTS_SYSTEM_PROMPT:
            return {"content": json.dumps({
                "target": "test fact",
                "scope": "single fact",
                "completion_criteria": ["Provide the sourced fact."],
            })}
        if system == agent_module.PLAN_PROMPT:
            return {"content": json.dumps(["web_read: https://example.com/weak"])}
        if system == agent_module.EVIDENCE_LEDGER_SYSTEM_PROMPT:
            ledger_calls += 1
            source_id = 1 if ledger_calls == 1 else 2
            return {"content": json.dumps({
                "answer_ready": True,
                "ledger": [{
                    "requirement": "test fact",
                    "proposed_claim": "The verified fact is 42.",
                    "source_ids": [source_id],
                    "support_level": "supported",
                    "can_use_in_answer": True,
                    "missing": "",
                }],
                "global_missing": [],
                "next_steps": [],
                "reason": "Ledger claims support.",
            })}
        if system == agent_module.EVIDENCE_CHALLENGE_SYSTEM_PROMPT:
            challenge_calls += 1
            if challenge_calls == 1:
                return {"content": json.dumps({
                    "answer_permitted": False,
                    "blocking_gaps": [{"requirement_index": 0, "text": "The first source is too thin for the claim."}],
                    "next_steps": ["web_read: https://example.com/strong"],
                    "reason": "Need a stronger source.",
                })}
            return {"content": json.dumps({
                "answer_permitted": True,
                "blocking_gaps": [],
                "next_steps": [],
                "reason": "The stronger source resolves the gap.",
            })}
        if system == agent_module.ANSWER_PROSE_SYSTEM_PROMPT:
            return {"content": "The verified fact is 42 [1]."}
        if system == agent_module.VERIFY_PROSE_SYSTEM_PROMPT:
            return {"content": "The verified fact is 42 [1]."}
        if system == agent_module.VERIFY_VERDICT_SYSTEM_PROMPT:
            return {"content": json.dumps({"coverage_complete": True, "missing": [], "notes": []})}
        if "After executing a batch of steps" in system:
            return {"content": json.dumps({"decision": "DONE", "reason": "continue to ledger"})}
        raise AssertionError(f"unexpected system prompt: {system[:80]}")

    async def fake_call_mcp_tool(name, args, session=None):
        assert name == "web_read"
        if args["url"].endswith("/weak"):
            return json.dumps({"url": args["url"], "title": "Weak", "text": "A short mention says 42."})
        return json.dumps({"url": args["url"], "title": "Strong", "text": "The verified fact is 42."})

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
        "task": "Find the test fact",
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
    assert "unless the user explicitly requested exhaustive coverage" in EVIDENCE_CHALLENGE_SYSTEM_PROMPT
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
        "evidence_audit": {"passed": False, "gaps": ["No fresh evidence path remains."]},
        "verification_result": {"insufficient_evidence": True},
        "final_answer": INSUFFICIENT_EVIDENCE_MESSAGE,
        "evidence_round": 1,
    }

    assert route_after_evidence_challenge(state) == "assimilate"


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
    call = agent_module._tool_call_from_step("web_fetch_json: https://example.com/api.json")

    assert call == {"function": {"name": "web_fetch_json", "arguments": {"url": "https://example.com/api.json"}}}


def test_strategy_plan_uses_memory_queries_as_search_steps():
    memory = agent_module._default_search_memory()
    memory["next_queries"] = ["specific source query", "another query"]

    steps = agent_module._strategy_plan_from_memory("daily rate for requested range", memory)

    # The refactored strategy emits search-only steps; the post-batch LLM decides
    # which result URLs to read. No deterministic placeholder scaffolding remains.
    assert steps == ["web_search: specific source query", "web_search: another query"]
    assert all(step.startswith("web_search: ") for step in steps)


def test_strategy_plan_does_not_fall_back_to_question_when_no_memory_queries():
    steps = agent_module._strategy_plan_from_memory("daily rate", agent_module._default_search_memory())

    assert steps == []


def test_execute_dedups_web_search_batch_and_runs_sequentially(monkeypatch):
    calls = []

    async def fake_call_mcp_tool(name, args, session=None):
        calls.append((name, args["query"]))
        return json.dumps({"count": 0, "sources": []})

    def fake_ollama_chat_json(*args, **kwargs):
        return json.dumps({"decision": "DONE", "reason": "batch behavior covered"})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(nodes_module, "_ollama_chat_json", fake_ollama_chat_json)
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

    update = asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE, fast_model="fast"))

    assert calls == [("web_search", "repeated query"), ("web_search", "different query")]
    assert [step["step"] for step in update["completed_steps"]] == [
        "web_search: repeated query",
        "web_search: different query",
    ]


def test_strategy_exhausts_when_no_fresh_search_direction(monkeypatch):
    def fake_chat(model, messages, tools=None, system="", temperature=0.3, json_mode=False, num_predict=32768, format_schema=None):
        return {"content": json.dumps({
            "search_hypothesis": "Try a different evidence path.",
            "failure_diagnosis": "Prior searches did not produce usable article-level evidence.",
            "mutation_dimension": "stop",
            "exhausted_direction": "general web search",
            "next_queries": [],
            "strategy_note": "No fresh search direction remains.",
        })}

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
    assert updated_memory["failure_diagnoses"] == ["Prior searches did not produce usable article-level evidence."]
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
            "arguments": {"task": "rates", "requirements": {"granularity": "day"}, "max_queries": 6},
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
    assert step["function"]["arguments"]["input_payload"]["source_url"] == "https://example.com/rates"
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
            "query_candidates": ["specific follow-up query", "specific follow-up query"],
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
    assert observation["summary"] == "The source partially discusses the requested item."
    assert observation["source_title"] == "Partial source"
    assert observation["publication_dates"] == ["2026-07-01"]
    assert observation["event_dates"] == ["2026-06-30"]
    assert observation["urls"] == ["https://example.org/source"]
    assert observation["errors"] == ["paywall notice"]
    assert observation["failure_diagnosis"] == "The source does not prove the required detail."


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

    update = asyncio.run(agent_module.evaluate_node(state, "fake", [], FOOTNOTE_PROFILE))
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
    memory = agent_module._record_observation(agent_module._default_search_memory(), observation, requirements)

    # Next-step planning is now an LLM decision; the deterministic memory bookkeeping
    # is what we assert: controller tags recorded, unhelpful source quarantined.
    assert observation["useful"] is False
    assert "search_better_sources" in memory["next_actions"]
    assert "source does not cover required detail" in memory["open_gaps"]
    assert "https://example.org/source" in memory["avoid_urls"]
    assert "example.org" in memory["bad_domains"]
    assert memory["query_candidates"] == ["specific required detail source"]
    assert memory["next_queries"] == []
    assert memory["observation_failure_diagnoses"] == ["The source lacks the required detail."]
    assert memory["observations"][0]["summary"] == "Partial source without the required detail."


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
    assert enriched[0]["tool_observation_summary"] == "Official page confirms the relevant fact."
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
    memory = agent_module._record_observation(agent_module._default_search_memory(), observation, requirements)

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
    tool-calling, treats its output as a source, and returns a grounded, verified answer."""
    from llmflow_search import prompts

    def fake_chat(model, messages, tools=None, system="", temperature=0.3, json_mode=False, num_predict=32768, format_schema=None):
        if system == prompts.REQUIREMENTS_SYSTEM_PROMPT:
            return {"content": json.dumps({
                "target": "capital of France",
                "required_coverage": "state the capital",
                "completion_criteria": ["Name the capital of France from a tool result."],
            })}
        if system == prompts.GENERIC_PLAN_PROMPT:
            return {"content": json.dumps({"steps": [{"tool": "step", "arg": "look up the capital of France"}]})}
        if tools:  # execute step → native function call against the foreign tool
            return {"content": "", "tool_calls": [
                {"function": {"name": "get_fact", "arguments": {"topic": "capital of france"}}}
            ]}
        if system == prompts.OBSERVATION_SYSTEM_PROMPT:
            return {"content": json.dumps({
                "useful": True, "structured": False, "has_rows": False, "dated": False,
                "source_quality": "secondary", "gaps": [], "next_action_tags": ["stop_and_answer"],
                "suggested_queries": [], "reason": "answered",
            })}
        if system == prompts.GENERIC_POST_BATCH_PROMPT:
            return {"content": json.dumps({"decision": "DONE", "next_steps": [], "reason": "covered"})}
        if system == prompts.ANSWER_PROSE_SYSTEM_PROMPT:
            return {"content": "The capital of France is Paris [1]."}
        if system == prompts.VERIFY_PROSE_SYSTEM_PROMPT:
            return {"content": "The capital of France is Paris [1]."}
        if system == prompts.VERIFY_VERDICT_SYSTEM_PROMPT:
            return {"content": json.dumps({"coverage_complete": True, "missing": [], "notes": []})}
        if system == prompts.EVIDENCE_LEDGER_SYSTEM_PROMPT:
            return {"content": json.dumps({
                "answer_ready": True,
                "ledger": [{
                    "requirement": "state the capital",
                    "proposed_claim": "The capital of France is Paris.",
                    "source_ids": [1],
                    "support_level": "supported",
                    "can_use_in_answer": True,
                    "missing": "",
                }],
                "global_missing": [],
                "next_steps": [],
                "reason": "The tool output supports the answer.",
            })}
        if system == prompts.EVIDENCE_CHALLENGE_SYSTEM_PROMPT:
            return {"content": json.dumps({
                "answer_permitted": True,
                "blocking_gaps": [],
                "next_steps": [],
                "reason": "No blocking evidence weakness.",
            })}
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

    tools = [{
        "type": "function",
        "function": {
            "name": "get_fact",
            "description": "Look up a short fact about a topic.",
            "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]},
        },
    }]
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
    sources = [{"title": "Source", "url": "https://example.com/a", "content": "Moscow news roundup."}]
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
            {"requirement_index": 5, "text": "Political news for the exact date is missing."},
        ],
        "next_steps": [],
        "reason": "Not all categories covered.",
    }

    result = _normalize_evidence_ledger_result(raw, sources, completion_criteria, "strict")

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

    raw_three = {"answer_ready": False, "ledger": [ledger_row(1), ledger_row(2), ledger_row(3)], "global_missing": [], "next_steps": []}
    result_three = _normalize_evidence_ledger_result(raw_three, sources, completion_criteria, "roundup")
    assert result_three["answer_ready"] is True

    raw_two = {"answer_ready": False, "ledger": [ledger_row(1), ledger_row(2)], "global_missing": [], "next_steps": []}
    result_two = _normalize_evidence_ledger_result(raw_two, sources, completion_criteria, "roundup")
    assert result_two["answer_ready"] is False


def test_evidence_ledger_dedup_stalled_routes_to_reextract(monkeypatch):
    def fake_chat(model, messages, tools=None, system="", temperature=0, num_predict=32768, format_schema=None, json_mode=False):
        return {"content": json.dumps({
            "answer_ready": False,
            "ledger": [],
            "global_missing": [],
            "next_steps": ["web_read: https://example.com/already-read"],
            "reason": "need more",
        })}

    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)

    state = {
        "task": "Find the requested fact",
        "requirements_result": {
            "answer_mode": "strict",
            "completion_criteria": ["Provide the requested fact."],
        },
        "candidate_sources": [{"title": "S", "url": "https://example.com/already-read", "content": "text"}],
        "sources": [],
        "completed_steps": [{"step": "web_read: https://example.com/already-read"}],
        "plan": [],
        "evidence_round": 0,
        "iteration": 0,
    }

    update = asyncio.run(evidence_ledger_node(state, "fake-model", [], FOOTNOTE_PROFILE))

    assert update["evidence_audit"]["dedup_stalled"] is True
    assert route_after_evidence_ledger(state | update) == "evidence_reextract"


def test_route_after_evidence_reextract_returns_challenge_when_recovered():
    assert route_after_evidence_reextract({"evidence_audit": {"passed": True}}) == "evidence_challenge"


def test_route_after_evidence_reextract_returns_assimilate_when_not_recovered():
    assert route_after_evidence_reextract({"evidence_audit": {"passed": False}}) == "assimilate"


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
        return json.dumps({
            "url": "https://example.com/news",
            "title": "Latest News",
            "text": "Listing page text with several headlines.",
            "is_listing": True,
            "links": [
                {"text": "City announces new metro line", "url": "https://example.com/a1"},
                {"text": "Mayor holds press conference", "url": "https://example.com/a2"},
                {"text": "Weather warning issued", "url": "https://example.com/a3"},
            ],
        })

    def fake_ollama_chat_json(*args, **kwargs):
        return json.dumps({"decision": "DONE", "reason": "listing page read"})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(nodes_module, "_ollama_chat_json", fake_ollama_chat_json)
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
        "search_memory": agent_module._default_search_memory(),
        "iteration": 0,
    }

    update = asyncio.run(execute_node(state, "main", [], FOOTNOTE_PROFILE, fast_model="fast"))

    assert update["plan"] == [
        "web_read: https://example.com/a1",
        "web_read: https://example.com/a2",
        "web_read: https://example.com/a3",
    ]
