"""Coverage of the non-core MCP tools: their results must become evidence,
their URLs must be registered, and the planner must be able to reach them."""

from __future__ import annotations

import asyncio
import json

from llmflow_search import llm, mcp_client
from llmflow_search import nodes as nodes_module
from llmflow_search.nodes import _drop_undiscovered_url_steps, execute_node
from llmflow_search.profiles import FOOTNOTE_PROFILE
from llmflow_search.search_memory import (
    _merge_search_memory,
    _strategy_plan_from_memory,
    _update_search_memory,
)
from llmflow_search.sources import _sources_from_tool_result
from llmflow_search.tool_steps import _tool_call_from_schema_step


def test_papers_search_records_with_abstracts_become_sources():
    payload = {
        "query": "scaling laws",
        "count": 2,
        "results": [
            {
                "title": "Scaling Laws for Neural Language Models",
                "url": "https://arxiv.org/abs/2001.08361",
                "snippet": "We study empirical scaling laws for language model performance "
                "on the cross-entropy loss, spanning seven orders of magnitude.",
                "published": "2020-01-23",
                "authors": ["Kaplan", "McCandlish"],
                "source": "arxiv",
                "identifiers": {"arxiv_id": "2001.08361"},
            },
            {
                "title": "Too short to cite",
                "url": "https://doi.org/10.1000/short",
                "snippet": "No abstract.",
                "source": "crossref",
            },
        ],
    }

    sources = _sources_from_tool_result("papers_search", json.dumps(payload))

    assert len(sources) == 1
    assert sources[0]["url"] == "https://arxiv.org/abs/2001.08361"
    assert sources[0]["kind"] == "paper"
    assert sources[0]["published"] == "2020-01-23"
    assert sources[0]["authors"] == ["Kaplan", "McCandlish"]
    assert sources[0]["identifiers"] == {"arxiv_id": "2001.08361"}
    # The abstract is labelled as index metadata in the source's type, not inside its
    # text: a warning prepended to the content read as a verdict on the whole evidence set.
    assert sources[0]["content"].startswith("We study empirical scaling laws")
    assert (
        sources[0]["source_quality"]["source_type"]
        == "arxiv paper record (abstract only)"
    )


def test_a_fetched_page_is_marked_as_full_text_beside_index_records():
    """With index records outnumbering it, one fetched page went unrecognised and the
    ledger reported having "only abstracts" for a page it had already read in full."""
    from llmflow_search.sources import _format_sources_for_llm

    records = _sources_from_tool_result(
        "encyclopedia_search",
        json.dumps(
            {
                "results": [
                    {
                        "title": "Entry",
                        "url": "https://ref.example/entry",
                        "snippet": "A summary sentence long enough to clear the record threshold "
                        "that this project applies to index results.",
                        "source": "wikipedia",
                    }
                ]
            }
        ),
    )
    page = _sources_from_tool_result(
        "web_read",
        json.dumps(
            {
                "url": "https://ref.example/entry/full",
                "title": "Entry",
                "text": "The full article body with the figures the question asks for.",
            }
        ),
    )

    text, shown = _format_sources_for_llm(records + page)

    assert len(shown) == 2
    assert "wikipedia encyclopedia record (abstract only)" in text
    assert "full text fetched" in text


def test_a_fetched_archive_record_is_not_labelled_as_index_metadata():
    payload = json.dumps(
        {
            "results": [
                {
                    "title": "Statistics 2024",
                    "url": "https://web.archive.org/web/20250101/https://gov.example/stats",
                    "text": "Annual figures for 2024: the recorded total was 41.2 units across all "
                    "regions, unchanged from the previous reporting period.",
                    "source": "wayback",
                }
            ]
        }
    )

    source = _sources_from_tool_result("archive_search", payload)[0]

    assert source["content"].startswith("Annual figures")
    assert "abstract only" not in source["source_quality"]["source_type"]


def test_github_and_encyclopedia_records_carry_their_own_kind():
    repo = json.dumps(
        {
            "results": [
                {
                    "title": "openai/tiktoken",
                    "url": "https://github.com/openai/tiktoken",
                    "snippet": "tiktoken is a fast BPE tokeniser for use with OpenAI's models, "
                    "written in Rust with Python bindings.",
                    "source": "github",
                }
            ]
        }
    )
    entity = json.dumps(
        {
            "results": [
                {
                    "title": "Euro",
                    "url": "https://en.wikipedia.org/wiki/Euro",
                    "snippet": "The euro is the official currency of 20 of the 27 member states "
                    "of the European Union, known collectively as the eurozone.",
                    "source": "wikipedia",
                }
            ]
        }
    )

    assert _sources_from_tool_result("github_search", repo)[0]["kind"] == "repository"
    assert (
        _sources_from_tool_result("encyclopedia_search", entity)[0]["kind"]
        == "encyclopedia"
    )


def test_web_archive_fetch_becomes_a_source_only_when_the_snapshot_was_read():
    read = json.dumps(
        {
            "url": "https://gov.example/statistics",
            "archived": True,
            "snapshot_url": "https://web.archive.org/web/20250101/https://gov.example/statistics",
            "title": "Statistics 2024",
            "text": "Annual figures: 2024 total was 41.2 units.",
            "published": "2025-01-01",
        }
    )
    unread = json.dumps(
        {
            "url": "https://gov.example/statistics",
            "archived": True,
            "snapshot_url": "https://web.archive.org/web/20250101/https://gov.example/statistics",
            "fetch_error": "empty snapshot body",
        }
    )

    sources = _sources_from_tool_result("web_archive_fetch", read)

    assert len(sources) == 1
    assert sources[0]["kind"] == "archive"
    assert sources[0]["url"].startswith("https://web.archive.org/")
    assert sources[0]["original_url"] == "https://gov.example/statistics"
    assert _sources_from_tool_result("web_archive_fetch", unread) == []


def test_web_crawl_yields_one_source_per_readable_page():
    payload = json.dumps(
        {
            "start_url": "https://example.com/docs",
            "pages_crawled": 3,
            "pages": [
                {
                    "url": "https://example.com/docs",
                    "title": "Docs",
                    "text": "Index of the docs.",
                },
                {
                    "url": "https://example.com/docs/a",
                    "title": "A",
                    "text": "Page A body.",
                },
                {
                    "url": "https://example.com/docs/b",
                    "error": "HTTP 404",
                    "text_length": 0,
                },
            ],
        }
    )

    sources = _sources_from_tool_result("web_crawl", payload)

    assert [source["url"] for source in sources] == [
        "https://example.com/docs",
        "https://example.com/docs/a",
    ]


def test_web_extract_is_attributed_to_the_current_browser_page():
    payload = json.dumps({"text": "Rate on 2026-08-05: 91.4 RUB per EUR."})

    with_context = _sources_from_tool_result(
        "web_extract", payload, {"browser_url": "https://bank.example/rates"}
    )
    without_context = _sources_from_tool_result("web_extract", payload)

    assert with_context[0]["url"] == "https://bank.example/rates"
    assert "91.4" in with_context[0]["content"]
    # No address means no provenance, so the text cannot be cited.
    assert without_context == []


def test_date_ranged_browser_tables_become_a_source_with_their_range():
    payload = json.dumps(
        {
            "date_range": {
                "start_date": "2026-07-01",
                "end_date": "2026-07-31",
                "submitted": True,
            },
            "url": "https://bank.example/history",
            "title": "History",
            "tables": [
                {
                    "columns": ["date", "rate"],
                    "rows": [{"date": "2026-07-01", "rate": "90.8"}],
                }
            ],
            "table_count": 1,
        }
    )

    sources = _sources_from_tool_result(
        "browser_extract_tables_for_date_range", payload
    )

    assert len(sources) == 1
    assert sources[0]["kind"] == "browser_table"
    assert "2026-07-31" in sources[0]["content"]


def test_detected_download_links_survive_the_invented_url_gate():
    memory = _update_search_memory(
        None,
        "web_detect_downloads",
        {"url": "https://gov.example/data"},
        json.dumps(
            {
                "url": "https://gov.example/data",
                "count": 1,
                "downloads": [
                    {
                        "url": "https://gov.example/data/2026.csv",
                        "text": "2026 dataset",
                        "extension": ".csv",
                    }
                ],
            }
        ),
    )

    assert "https://gov.example/data/2026.csv" in memory["discovered_urls"]

    kept, invented = _drop_undiscovered_url_steps(
        [
            "web_parse_file: https://gov.example/data/2026.csv",
            "web_parse_file: https://gov.example/data/guessed.csv",
        ],
        set(memory["discovered_urls"]),
    )

    assert kept == ["web_parse_file: https://gov.example/data/2026.csv"]
    assert invented == ["web_parse_file: https://gov.example/data/guessed.csv"]


def test_subject_index_results_are_registered_as_discovered_urls():
    memory = _update_search_memory(
        None,
        "papers_search",
        {"query": "scaling laws"},
        json.dumps(
            {
                "count": 1,
                "results": [
                    {"title": "Scaling Laws", "url": "https://arxiv.org/abs/2001.08361"}
                ],
            }
        ),
    )

    assert memory["discovered_urls"] == ["https://arxiv.org/abs/2001.08361"]
    assert (
        memory["discovered_titles"]["https://arxiv.org/abs/2001.08361"]
        == "Scaling Laws"
    )
    assert memory["attempted_queries"] == ["scaling laws"]

    archive = _update_search_memory(
        None,
        "archive_search",
        {"url": "https://gov.example/gone"},
        json.dumps(
            {"count": 1, "results": [{"url": "https://web.archive.org/web/2020/gone"}]}
        ),
    )

    # archive_search is keyed by URL, not by a query string.
    assert archive["attempted_queries"] == ["https://gov.example/gone"]
    assert archive["discovered_urls"] == ["https://web.archive.org/web/2020/gone"]


def test_browser_navigation_records_the_page_web_extract_will_be_attributed_to():
    memory = _update_search_memory(
        None,
        "web_navigate",
        {"url": "https://bank.example/rates"},
        json.dumps({"url": "https://bank.example/rates", "title": "Rates"}),
    )

    assert memory["browser_url"] == "https://bank.example/rates"
    assert _merge_search_memory(memory)["browser_url"] == "https://bank.example/rates"


def test_strategy_plan_keeps_a_changed_access_path_and_defaults_to_web_search():
    memory = _merge_search_memory(
        {
            "next_queries": [
                "papers_search: transformer scaling laws",
                "euro rate august 2026",
            ]
        }
    )

    steps = _strategy_plan_from_memory("q", memory, {"web_search", "papers_search"})

    assert steps == [
        "papers_search: transformer scaling laws",
        "web_search: euro rate august 2026",
    ]


def test_strategy_plan_falls_back_to_web_search_for_an_unavailable_tool():
    memory = _merge_search_memory({"next_queries": ["papers_search: scaling laws"]})

    steps = _strategy_plan_from_memory("q", memory, {"web_search"})

    assert steps == ["web_search: papers_search: scaling laws"]


_PAPERS_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "papers_search",
        "description": "Search scientific publications.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "num": {"type": "integer"}},
            "required": ["query"],
        },
    },
}


def test_a_subject_index_step_runs_and_reaches_the_ledger_as_a_source(monkeypatch):
    """End to end for one non-core tool: planner step → MCP call → candidate source."""
    calls = []

    async def fake_call_mcp_tool(name, args, session=None):
        calls.append((name, args))
        return json.dumps(
            {
                "query": args["query"],
                "count": 1,
                "results": [
                    {
                        "title": "Scaling Laws for Neural Language Models",
                        "url": "https://arxiv.org/abs/2001.08361",
                        "snippet": "We study empirical scaling laws for language model performance "
                        "on the cross-entropy loss, spanning seven orders of magnitude.",
                        "published": "2020-01-23",
                        "source": "arxiv",
                    }
                ],
            }
        )

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "DONE", "reason": "paper found"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    state = {
        "task": "Find the scaling-laws paper",
        "requirements_result": {},
        "plan": ["papers_search: scaling laws for neural language models"],
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "search_memory": _merge_search_memory(None),
        "iteration": 0,
    }

    update = asyncio.run(
        execute_node(state, "main", [_PAPERS_SEARCH_TOOL], FOOTNOTE_PROFILE)
    )

    # Resolved from the live schema — no hard-coded parser entry for this tool.
    assert calls == [
        ("papers_search", {"query": "scaling laws for neural language models"})
    ]
    assert [source["url"] for source in update["candidate_sources"]] == [
        "https://arxiv.org/abs/2001.08361"
    ]
    assert update["search_memory"]["discovered_urls"] == [
        "https://arxiv.org/abs/2001.08361"
    ]


# ── step parsing: the form a model actually emits ──

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "lang": {"type": "string"},
                "num": {"type": "integer"},
                "provider": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}

_WEB_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "web_read",
        "description": "Read a page.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "lang": {"type": "string"},
                "use_cache": {"type": "boolean"},
            },
            "required": ["url"],
        },
    },
}

_PARSE_TOOLS = [_WEB_SEARCH_TOOL, _WEB_READ_TOOL]


def test_named_arguments_are_resolved_against_the_schema():
    """The whole string used to be swallowed as the query, so every search ran on
    the literal text "query='...' lang='en' num=10" and returned nothing usable."""
    call = _tool_call_from_schema_step(
        "web_search: query='Spain August 2026 events' lang='en' num=10 provider='brave'",
        _PARSE_TOOLS,
    )

    assert call["function"]["arguments"] == {
        "query": "Spain August 2026 events",
        "lang": "en",
        "num": 10,
        "provider": "brave",
    }


def test_named_arguments_work_comma_separated_and_with_booleans():
    call = _tool_call_from_schema_step(
        "web_read: url=https://example.com/a, lang=en, use_cache=false", _PARSE_TOOLS
    )

    assert call["function"]["arguments"] == {
        "url": "https://example.com/a",
        "lang": "en",
        "use_cache": False,
    }


def test_plain_value_and_json_object_forms_still_work():
    plain = _tool_call_from_schema_step("web_search: madrid august 2026", _PARSE_TOOLS)
    obj = _tool_call_from_schema_step(
        'web_search: {"query": "madrid august 2026", "num": 5}', _PARSE_TOOLS
    )
    url = _tool_call_from_schema_step(
        "web_read: https://example.com/a?x=1", _PARSE_TOOLS
    )

    assert plain["function"]["arguments"] == {"query": "madrid august 2026"}
    assert obj["function"]["arguments"] == {"query": "madrid august 2026", "num": 5}
    assert url["function"]["arguments"] == {"url": "https://example.com/a?x=1"}


def test_a_search_operator_is_a_query_not_a_named_argument():
    # "site" is not a parameter of web_search, so "site=..." / "site:..." is part of
    # the query the user wants, not a mis-typed argument.
    for step in (
        "web_search: site=example.com spain news",
        "web_search: site:example.com spain news",
    ):
        call = _tool_call_from_schema_step(step, _PARSE_TOOLS)
        assert call["function"]["arguments"]["query"] == step.split(": ", 1)[1]


def test_a_bare_url_is_never_mistaken_for_named_arguments():
    # "https://..." opens with "https:", which is the same shape as "url:".
    call = _tool_call_from_schema_step(
        "web_read: https://example.com/a?x=1", _PARSE_TOOLS
    )
    assert call["function"]["arguments"] == {"url": "https://example.com/a?x=1"}


def test_a_spurious_tool_head_is_stripped_before_parsing():
    # The planner once emitted "tool:web_extract_tables:https://..." and the step was
    # dropped as "unknown tool" — the leading "tool:" is scaffolding, not a tool name.
    from llmflow_search.tool_steps import _strip_spurious_head, _tool_call_from_step

    step = "tool:web_extract_tables:https://www.sravni.ru/valjuty/info/kurs-eur-05-2026"
    call = _tool_call_from_schema_step(step, _PARSE_TOOLS)
    assert call is None  # web_extract_tables is not among _PARSE_TOOLS
    call = _tool_call_from_step(step)
    assert call["function"]["name"] == "web_extract_tables"
    assert call["function"]["arguments"] == {
        "url": "https://www.sravni.ru/valjuty/info/kurs-eur-05-2026"
    }
    # A plain step is untouched, and so is an identity: the wrapped and the plain form
    # of the same call dedup as one.
    assert (
        _strip_spurious_head("web_search: spain august 2026")
        == "web_search: spain august 2026"
    )
    call = _tool_call_from_schema_step(
        "tool:web_read: https://example.com/a?x=1", _PARSE_TOOLS
    )
    assert call["function"]["arguments"] == {"url": "https://example.com/a?x=1"}


def test_colon_separated_named_arguments_are_resolved():
    """The model switched from "key=value" to "key: value" and the whole string was
    swallowed as the query again — searches ran on the literal text."""
    call = _tool_call_from_schema_step(
        "web_search: query: 'best local LLM August 2026' lang:en num:10", _PARSE_TOOLS
    )

    assert call["function"]["arguments"] == {
        "query": "best local LLM August 2026",
        "lang": "en",
        "num": 10,
    }


def test_an_unquoted_multi_word_value_survives_intact():
    call = _tool_call_from_schema_step(
        "web_search: query: python 3.14 release notes", _PARSE_TOOLS
    )

    assert call["function"]["arguments"] == {"query": "python 3.14 release notes"}


def test_a_named_argument_with_no_value_is_refused():
    assert _tool_call_from_schema_step("web_search: query:", _PARSE_TOOLS) is None


def test_the_same_call_written_two_ways_has_one_identity():
    from llmflow_search.tool_steps import _step_identity

    plain = _step_identity("web_search: best local LLM August 2026", _PARSE_TOOLS)
    named = _step_identity(
        "web_search: query: 'best local LLM August 2026' num:10", _PARSE_TOOLS
    )
    json_form = _step_identity(
        'web_search: {"query": "best local LLM August 2026", "lang": "en"}',
        _PARSE_TOOLS,
    )
    other = _step_identity("web_search: something else entirely", _PARSE_TOOLS)

    assert plain == named == json_form
    assert other != plain


# ── pacing: throttled backends get one request at a time, a few per round ──


def _tool(name, params, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": params,
                "required": required,
            },
        },
    }


_PACING_TOOLS = [
    _tool(
        "web_search",
        {
            "query": {"type": "string"},
            "lang": {"type": "string"},
            "num": {"type": "integer"},
        },
        ["query"],
    ),
    _tool("web_search_recent", {"query": {"type": "string"}}, ["query"]),
    _tool("web_read", {"url": {"type": "string"}}, ["url"]),
]


def _throttled_state(plan, task="q"):
    # Read steps name pages the run has seen; a plan is filtered against that set, so a
    # fixture that skips discovery has to say which addresses were discovered.
    memory = _merge_search_memory(None)
    memory["discovered_urls"] = [
        step.split(": ", 1)[1] for step in plan if step.startswith("web_read: ")
    ]
    return {
        "task": task,
        "requirements_result": {},
        "plan": plan,
        "completed_steps": [],
        "scratchpad": "",
        "candidate_sources": [],
        "admissible_sources": [],
        "sources": [],
        "search_memory": memory,
        "iteration": 0,
    }


def test_a_long_search_batch_is_spread_across_rounds(monkeypatch):
    """Six queued searches used to fire in one go; each one fans out to four
    engines inside the server, so that is two dozen upstream requests at once."""
    order = []

    async def fake_call_mcp_tool(name, args, session=None):
        order.append(args.get("query"))
        return json.dumps({"count": 0, "results": []})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "CONTINUE", "reason": "keep going"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})
    monkeypatch.setattr(nodes_module, "group_batch_cap", lambda group: 3)

    plan = [f"web_search: query {i}" for i in range(6)]
    update = asyncio.run(
        execute_node(_throttled_state(plan), "main", _PACING_TOOLS, FOOTNOTE_PROFILE)
    )

    assert order == ["query 0", "query 1", "query 2"]
    assert update["plan"] == [f"web_search: query {i}" for i in range(3, 6)]


def test_search_batches_auto_queue_page_reads_before_more_searching(monkeypatch):
    async def fake_call_mcp_tool(name, args, session=None):
        assert name == "web_search"
        return json.dumps(
            {
                "count": 2,
                "results": [
                    {"url": "https://example.com/a", "title": "A"},
                    {"url": "https://example.com/b", "title": "B"},
                ],
            }
        )

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "CONTINUE", "reason": "keep searching"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    update = asyncio.run(
        execute_node(
            _throttled_state(["web_search: first query", "web_search: second query"]),
            "main",
            _PACING_TOOLS,
            FOOTNOTE_PROFILE,
        )
    )

    assert update["search_memory"]["discovered_urls"] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert update["plan"][:2] == [
        "web_read: https://example.com/a",
        "web_read: https://example.com/b",
    ]


def test_every_throttled_family_runs_one_at_a_time(monkeypatch):
    """Only web_search was forced sequential; web_search_recent and the subject
    indexes ran fully in parallel and emptied the rate-limit budget at once."""
    concurrent = {"now": 0, "peak": 0}

    async def fake_call_mcp_tool(name, args, session=None):
        concurrent["now"] += 1
        concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
        await asyncio.sleep(0.01)
        concurrent["now"] -= 1
        return json.dumps({"count": 0, "results": []})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "CONTINUE", "reason": "x"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    plan = [f"web_search_recent: query {i}" for i in range(3)]
    asyncio.run(
        execute_node(_throttled_state(plan), "main", _PACING_TOOLS, FOOTNOTE_PROFILE)
    )

    assert concurrent["peak"] == 1


def test_unthrottled_tools_still_run_in_parallel(monkeypatch):
    concurrent = {"now": 0, "peak": 0}

    async def fake_call_mcp_tool(name, args, session=None):
        concurrent["now"] += 1
        concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
        await asyncio.sleep(0.01)
        concurrent["now"] -= 1
        return json.dumps({"url": args.get("url"), "text": "body", "title": "t"})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "DONE", "reason": "x"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    plan = [f"web_read: https://example.com/{i}" for i in range(3)]
    asyncio.run(
        execute_node(_throttled_state(plan), "main", _PACING_TOOLS, FOOTNOTE_PROFILE)
    )

    assert concurrent["peak"] == 3  # reading pages is not rate-limited


def test_the_search_budget_stops_a_question_from_draining_the_quota(monkeypatch):
    """A run that keeps re-searching without finding sources burned 41 calls on one
    question; on a keyed provider that is a month's free tier in a few questions."""
    calls = []

    async def fake_call_mcp_tool(name, args, session=None):
        calls.append(args.get("query"))
        return json.dumps({"count": 0, "results": []})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "CONTINUE", "reason": "x"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})
    monkeypatch.setattr(nodes_module, "MAX_SEARCH_CALLS_PER_QUESTION", 3)
    monkeypatch.setattr(nodes_module, "group_batch_cap", lambda group: 2)

    state = _throttled_state([f"web_search: query {i}" for i in range(4)])
    first = asyncio.run(execute_node(state, "main", _PACING_TOOLS, FOOTNOTE_PROFILE))
    assert first["search_memory"]["search_calls"] == 2

    state = dict(state, **first)
    second = asyncio.run(execute_node(state, "main", _PACING_TOOLS, FOOTNOTE_PROFILE))
    assert second["search_memory"]["search_calls"] == 3  # budget clamps the last batch

    # A later round proposes another search; the budget is spent, so it is refused.
    state = {**state, **second, "plan": ["web_search: one more"]}
    third = asyncio.run(execute_node(state, "main", _PACING_TOOLS, FOOTNOTE_PROFILE))

    assert len(calls) == 3  # "one more" never went out
    assert third["search_memory"]["search_exhausted"] is True


def test_the_budget_does_not_block_reading_what_was_already_found(monkeypatch):
    async def fake_call_mcp_tool(name, args, session=None):
        return json.dumps({"url": args.get("url"), "text": "body", "title": "t"})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "DONE", "reason": "x"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})
    monkeypatch.setattr(nodes_module, "MAX_SEARCH_CALLS_PER_QUESTION", 0)

    state = _throttled_state(["web_search: spent", "web_read: https://example.com/a"])
    update = asyncio.run(execute_node(state, "main", _PACING_TOOLS, FOOTNOTE_PROFILE))

    # Searching stops; fetching an already-discovered page costs nothing and continues.
    assert update["plan"] == ["web_read: https://example.com/a"]


# ── source budget: every fetched page must reach the ledger ──


def test_every_fetched_source_is_shown_to_the_model(monkeypatch):
    """A fixed per-source cap exhausted the total budget partway down the list:
    25 pages were fetched and only 9 reached the ledger, so the rest of the
    rate-limit budget spent on fetching them bought nothing."""
    from llmflow_search.sources import _format_sources_for_llm

    sources = [
        {"title": f"t{i}", "url": f"https://e{i}.example", "content": "x" * 12000}
        for i in range(25)
    ]

    text, shown = _format_sources_for_llm(sources)

    assert len(shown) == 25
    assert len(text) <= 100_000


def test_a_short_source_list_still_gets_long_excerpts():
    from llmflow_search.sources import _format_sources_for_llm

    sources = [{"title": "t", "url": "https://e.example", "content": "x" * 30000}]

    text, shown = _format_sources_for_llm(sources)

    assert shown == {1}
    assert len(text) > 20000  # one source keeps the full per-source cap


def test_excerpts_never_shrink_below_a_usable_floor():
    from llmflow_search.sources import MIN_SOURCE_EXCERPT_CHARS, _source_excerpt_budget

    assert _source_excerpt_budget(500) == MIN_SOURCE_EXCERPT_CHARS
    assert _source_excerpt_budget(0) > MIN_SOURCE_EXCERPT_CHARS


def test_a_reworded_repeat_of_an_executed_step_is_dropped(monkeypatch):
    """The ledger re-proposed the same search with different wording each round;
    a string compare let it through and the run ground on for 40 steps."""
    calls = []

    async def fake_call_mcp_tool(name, args, session=None):
        calls.append(args.get("query"))
        return json.dumps({"count": 0, "results": []})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps(
            {
                "decision": "NEXT",
                # Same call as the one just executed, only written with named arguments.
                "next_steps": ["web_search: query: 'best local LLM' lang:en num:10"],
                "reason": "x",
            }
        ),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    state = _throttled_state(["web_search: best local LLM"])
    update = asyncio.run(execute_node(state, "main", _PACING_TOOLS, FOOTNOTE_PROFILE))

    assert calls == ["best local LLM"]
    assert update["plan"] == []


def test_page_reading_is_capped_instead_of_bursting(monkeypatch):
    concurrent = {"now": 0, "peak": 0}

    async def fake_call_mcp_tool(name, args, session=None):
        concurrent["now"] += 1
        concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
        await asyncio.sleep(0.01)
        concurrent["now"] -= 1
        return json.dumps({"url": args.get("url"), "text": "body", "title": "t"})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps({"decision": "DONE", "reason": "x"}),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})
    monkeypatch.setattr(nodes_module, "MAX_PARALLEL_FETCHES", 5)

    plan = [f"web_read: https://e{i}.example/a" for i in range(12)]
    asyncio.run(
        execute_node(_throttled_state(plan), "main", _PACING_TOOLS, FOOTNOTE_PROFILE)
    )

    assert concurrent["peak"] == 5  # twelve sites at once was the old behaviour


def test_continue_with_an_empty_plan_is_not_a_silent_dead_end(monkeypatch):
    """CONTINUE means "run the rest of the plan". With nothing left it fetched nothing
    and ended the round — the model picks it after a search whose snippets look like
    the answer, which is exactly when a page still has to be read."""

    async def fake_call_mcp_tool(name, args, session=None):
        return json.dumps(
            {
                "count": 1,
                "results": [
                    {"title": "Discography", "url": "https://ref.example/disco"}
                ],
            }
        )

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps(
            {
                "reason": "the snippets already say 14",
                "next_steps": [],
                "decision": "CONTINUE",
            }
        ),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    state = _throttled_state(["web_search: artist discography"])
    update = asyncio.run(execute_node(state, "main", _PACING_TOOLS, FOOTNOTE_PROFILE))

    # Coerced to NEXT: a round that fetched no source cannot be allowed to end quietly.
    assert update["search_memory"]["discovered_urls"] == ["https://ref.example/disco"]
    assert update["sources"] == []


def test_a_fetched_page_replaces_the_index_record_for_the_same_url():
    """An index record and the page it points at share a URL. First-wins let the
    one-line abstract squat on the address, so the page fetched afterwards — the only
    version that answers anything — was dropped as a duplicate."""
    from llmflow_search.sources import _merge_sources

    record = _sources_from_tool_result(
        "encyclopedia_search",
        json.dumps(
            {
                "results": [
                    {
                        "title": "Entry",
                        "url": "https://ref.example/entry",
                        "snippet": "A one-line summary that is long enough to clear the record threshold.",
                        "source": "wikipedia",
                    }
                ]
            }
        ),
    )
    page = _sources_from_tool_result(
        "web_read",
        json.dumps(
            {
                "url": "https://ref.example/entry",
                "title": "Entry",
                "text": "The full body, with the figures the question actually asks for.",
            }
        ),
    )

    merged = _merge_sources(record, page)

    assert len(merged) == 1
    assert merged[0]["kind"] == "page"
    assert "figures the question actually asks for" in merged[0]["content"]


def test_a_thinner_duplicate_never_displaces_what_was_already_fetched():
    from llmflow_search.sources import _merge_sources

    page = _sources_from_tool_result(
        "web_read",
        json.dumps(
            {
                "url": "https://ref.example/entry",
                "title": "Entry",
                "text": "The full body, with the figures the question actually asks for.",
            }
        ),
    )
    record = _sources_from_tool_result(
        "encyclopedia_search",
        json.dumps(
            {
                "results": [
                    {
                        "title": "Entry",
                        "url": "https://ref.example/entry",
                        "snippet": "A one-line summary that is long enough to clear the record threshold.",
                        "source": "wikipedia",
                    }
                ]
            }
        ),
    )

    merged = _merge_sources(page, record)

    assert len(merged) == 1
    assert merged[0]["kind"] == "page"


def test_merging_keeps_distinct_urls_and_their_order():
    from llmflow_search.sources import _merge_sources

    first = [
        {"title": "a", "url": "https://a.example", "content": "x" * 100, "kind": "page"}
    ]
    second = [
        {
            "title": "b",
            "url": "https://b.example",
            "content": "y" * 100,
            "kind": "page",
        },
        {
            "title": "c",
            "url": "https://c.example",
            "content": "z" * 100,
            "kind": "page",
        },
    ]

    merged = _merge_sources(first, second)

    assert [s["url"] for s in merged] == [
        "https://a.example",
        "https://b.example",
        "https://c.example",
    ]


def test_an_api_endpoint_may_be_constructed_without_prior_discovery(monkeypatch):
    """The plan prompt tells the planner to hit a known JSON endpoint directly rather
    than search for it, so gating those the same way would forbid documented behaviour.
    Only tools whose argument must have been pointed at are held to discovery."""
    called = []

    async def fake_call_mcp_tool(name, args, session=None):
        called.append((name, args.get("url")))
        return json.dumps({"url": args.get("url"), "json": {"value": 1}})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps(
            {"reason": "x", "next_steps": [], "decision": "DONE"}
        ),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    tools = _PACING_TOOLS + [
        _tool("web_fetch_json", {"url": {"type": "string"}}, ["url"])
    ]
    state = _throttled_state(["web_fetch_json: https://api.example/v1/series"])
    asyncio.run(execute_node(state, "main", tools, FOOTNOTE_PROFILE))

    assert called == [("web_fetch_json", "https://api.example/v1/series")]


def test_a_planned_read_of_an_address_no_one_produced_is_dropped(monkeypatch):
    called = []

    async def fake_call_mcp_tool(name, args, session=None):
        called.append(name)
        return json.dumps({"count": 0, "results": []})

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps(
            {"reason": "x", "next_steps": [], "decision": "DONE"}
        ),
    )
    monkeypatch.setattr(llm, "_ollama_chat", lambda *a, **k: {"content": ""})

    state = _throttled_state(["web_search: a topic"], task="a topic")
    state["plan"] = ["web_extract_tables: https://vendor.example/historical-data/"]
    update = asyncio.run(execute_node(state, "main", _PACING_TOOLS, FOOTNOTE_PROFILE))

    assert called == []
    assert update["plan"] == []


# ── a bot-protection page is not evidence ──


def test_an_interstitial_never_becomes_a_source():
    """It supports nothing, but as a source it made the run look like it had one: the
    controller reported DONE, the ledger found nothing, and the round repeated."""
    blocked = json.dumps(
        {
            "url": "https://guarded.example/data",
            "title": "Just a moment...",
            "text": "Just a moment...\nVerify you are human by completing the action below.",
        }
    )

    assert _sources_from_tool_result("web_read", blocked) == []


def test_an_interstitial_inside_deep_search_output_is_dropped():
    payload = json.dumps(
        {
            "sources": [
                {"num": 1, "url": "https://guarded.example/a", "title": "Guarded"},
                {"num": 2, "url": "https://real.example/b", "title": "Real"},
            ],
            "context": (
                "[1] Guarded\nChecking your browser before accessing the site. Please enable "
                "JavaScript and cookies to continue.\n"
                "[2] Real\nThe recorded figure for the period was 41.2 units, published by the "
                "statistics office in its quarterly bulletin.\n"
            ),
        }
    )

    sources = _sources_from_tool_result("web_deep_search", payload)

    assert [s["url"] for s in sources] == ["https://real.example/b"]


def test_a_long_article_mentioning_captchas_is_still_a_source():
    """The marker check is bounded by length on purpose — an article about bot
    protection is a legitimate source and must not be thrown away."""
    article = "A study of recaptcha adoption across the web. " + (
        "Detailed analysis follows. " * 120
    )
    payload = json.dumps(
        {"url": "https://real.example/study", "title": "Study", "text": article}
    )

    sources = _sources_from_tool_result("web_read", payload)

    assert len(sources) == 1
    assert sources[0]["url"] == "https://real.example/study"


def test_a_step_that_names_no_live_tool_is_skipped_not_improvised(monkeypatch):
    """The ledger's JSON template showed ["tool: argument"], and the placeholder was
    copied literally. Unresolvable, the step was handed to the model to improvise, which
    called a browser tool with no arguments at all."""
    called = []
    improvised = []

    async def fake_call_mcp_tool(name, args, session=None):
        called.append((name, args))
        return json.dumps({"ok": True})

    def fake_chat(*a, **k):
        improvised.append(k.get("system"))
        return {"tool_calls": [{"function": {"name": "web_read", "arguments": {}}}]}

    monkeypatch.setattr(mcp_client, "_call_mcp_tool", fake_call_mcp_tool)
    monkeypatch.setattr(llm, "_ollama_chat", fake_chat)
    monkeypatch.setattr(
        llm,
        "_ollama_chat_schema",
        lambda *a, **k: json.dumps(
            {"reason": "x", "next_steps": [], "decision": "DONE"}
        ),
    )

    state = _throttled_state(["tool: web_navigate"])
    asyncio.run(execute_node(state, "main", _PACING_TOOLS, FOOTNOTE_PROFILE))

    assert called == []
    assert improvised == []
