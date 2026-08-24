import asyncio
import json
from types import SimpleNamespace

import pytest

from llmflow_search import mcp_client


def test_search_request_delay_applies_across_calls(monkeypatch):
    sleeps = []
    clock = iter([100.0, 101.0, 103.0])

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(
        mcp_client,
        "SEARCH_GROUP_DELAY_SECONDS",
        {"scraper": 3.0, "api": 3.0, "archive": 3.0},
    )
    monkeypatch.setattr(mcp_client, "_random", lambda: 0.0)  # pin jitter
    monkeypatch.setattr(mcp_client, "_search_request_locks", {})
    monkeypatch.setattr(mcp_client, "_last_search_request_started_at", {})
    monkeypatch.setattr(mcp_client, "_monotonic", lambda: next(clock))
    monkeypatch.setattr(mcp_client.asyncio, "sleep", fake_sleep)

    async def run():
        await mcp_client._wait_for_search_request_slot("web_search")
        await mcp_client._wait_for_search_request_slot("web_deep_search")

    asyncio.run(run())

    assert sleeps == [2.0]
    assert mcp_client._last_search_request_started_at == {"scraper": 103.0}


def test_search_delay_is_tracked_per_backend_group(monkeypatch):
    """A scraper pause must not be charged to an API-backed search tool."""
    sleeps = []
    clock = iter([100.0, 100.2, 100.7, 103.0])

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(
        mcp_client,
        "SEARCH_GROUP_DELAY_SECONDS",
        {"scraper": 3.0, "api": 3.0, "archive": 3.0},
    )
    monkeypatch.setattr(mcp_client, "_random", lambda: 0.0)  # pin jitter
    monkeypatch.setattr(mcp_client, "_search_request_locks", {})
    monkeypatch.setattr(mcp_client, "_last_search_request_started_at", {})
    monkeypatch.setattr(mcp_client, "_monotonic", lambda: next(clock))
    monkeypatch.setattr(mcp_client.asyncio, "sleep", fake_sleep)

    async def run():
        await mcp_client._wait_for_search_request_slot("web_search")
        await mcp_client._wait_for_search_request_slot("papers_search")
        await mcp_client._wait_for_search_request_slot("github_search")

    asyncio.run(run())

    # papers_search opens its own group and waits for nothing the scraper did;
    # github_search shares that group and is throttled against papers_search.
    assert sleeps == [2.5]
    assert mcp_client._last_search_request_started_at == {
        "scraper": 100.0,
        "api": 103.0,
    }


def test_non_search_tool_has_no_delay(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(
        mcp_client,
        "SEARCH_GROUP_DELAY_SECONDS",
        {"scraper": 3.0, "api": 3.0, "archive": 3.0},
    )
    monkeypatch.setattr(mcp_client, "_random", lambda: 0.0)  # pin jitter
    monkeypatch.setattr(mcp_client.asyncio, "sleep", fake_sleep)

    asyncio.run(mcp_client._wait_for_search_request_slot("web_read"))

    assert sleeps == []


def test_tool_catalog_preserves_every_live_tool_and_required_parameters():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "papers_search",
                "description": "Search scientific publications.\nUses Crossref.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
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

    catalog = mcp_client._format_tool_catalog(tools)

    assert "papers_search(query*:string, limit:integer)" in catalog
    assert "Search scientific publications. Uses Crossref." in catalog
    assert "web_screenshot()" in catalog


def test_mcp_result_text_keeps_text_and_structured_content():
    result = SimpleNamespace(
        content=[SimpleNamespace(text="Human-readable result")],
        structuredContent={"rows": [{"value": 42}]},
        isError=False,
    )

    text = mcp_client._mcp_result_text(result)

    assert "Human-readable result" in text
    assert json.dumps(result.structuredContent, ensure_ascii=False) in text


def test_mcp_result_text_raises_server_reported_errors():
    result = SimpleNamespace(
        content=[SimpleNamespace(text="invalid date range")],
        structuredContent=None,
        isError=True,
    )

    with pytest.raises(RuntimeError, match="invalid date range"):
        mcp_client._mcp_result_text(result)


def test_jitter_widens_the_interval_so_calls_are_not_metronomic(monkeypatch):
    sleeps = []
    clock = iter([100.0, 100.0, 100.0])

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(mcp_client, "SEARCH_GROUP_DELAY_SECONDS", {"scraper": 10.0})
    monkeypatch.setattr(mcp_client, "SEARCH_DELAY_JITTER", 0.5)
    monkeypatch.setattr(mcp_client, "_random", lambda: 1.0)  # worst-case draw
    monkeypatch.setattr(mcp_client, "_search_request_locks", {})
    monkeypatch.setattr(mcp_client, "_last_search_request_started_at", {})
    monkeypatch.setattr(mcp_client, "_monotonic", lambda: next(clock))
    monkeypatch.setattr(mcp_client.asyncio, "sleep", fake_sleep)

    async def run():
        await mcp_client._wait_for_search_request_slot("web_search")
        await mcp_client._wait_for_search_request_slot("web_search")

    asyncio.run(run())

    assert sleeps == [15.0]  # 10s interval + 50% jitter
