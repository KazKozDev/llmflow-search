import asyncio

from llmflow_search import mcp_client


def test_search_request_delay_applies_across_calls(monkeypatch):
    sleeps = []
    clock = iter([100.0, 101.0, 103.0])

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(mcp_client, "SEARCH_REQUEST_DELAY_SECONDS", 3.0)
    monkeypatch.setattr(mcp_client, "_search_request_lock", asyncio.Lock())
    monkeypatch.setattr(mcp_client, "_last_search_request_started_at", 0.0)
    monkeypatch.setattr(mcp_client, "_monotonic", lambda: next(clock))
    monkeypatch.setattr(mcp_client.asyncio, "sleep", fake_sleep)

    async def run():
        await mcp_client._wait_for_search_request_slot("web_search")
        await mcp_client._wait_for_search_request_slot("web_deep_search")

    asyncio.run(run())

    assert sleeps == [2.0]
    assert mcp_client._last_search_request_started_at == 103.0


def test_non_search_tool_has_no_delay(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(mcp_client, "SEARCH_REQUEST_DELAY_SECONDS", 3.0)
    monkeypatch.setattr(mcp_client.asyncio, "sleep", fake_sleep)

    asyncio.run(mcp_client._wait_for_search_request_slot("web_read"))

    assert sleeps == []
