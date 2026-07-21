import asyncio

import httpx
import pytest
from pydantic import BaseModel

from deep_research.llm import LLMError, LLMResponseError, OllamaClient, StaticLLM


class _Schema(BaseModel):
    ok: bool


def _client(**kwargs) -> OllamaClient:
    return OllamaClient("http://localhost:11434", 5, max_retries=kwargs.pop("max_retries", 2), **kwargs)


def test_complete_returns_message_content(mock_transport) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={"message": {"content": " hello "}})

    mock_transport(handler)
    result = asyncio.run(_client().complete(model="m", system="s", user="u"))

    assert result == "hello"


def test_complete_raises_on_empty_message(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, json={"message": {"content": "   "}}))

    with pytest.raises(LLMResponseError, match="empty message"):
        asyncio.run(_client().complete(model="m", system="s", user="u"))


def test_complete_json_parses_schema(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, json={"message": {"content": '{"ok": true}'}}))

    result = asyncio.run(_client().complete_json(model="m", system="s", user="u", schema=_Schema))

    assert result.ok is True


def test_complete_json_raises_on_schema_mismatch(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, json={"message": {"content": '{"unexpected": 1}'}}))

    with pytest.raises(LLMResponseError, match="did not match"):
        asyncio.run(_client().complete_json(model="m", system="s", user="u", schema=_Schema))


def test_complete_json_raises_when_content_missing(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, json={"message": {}}))

    with pytest.raises(LLMResponseError, match="did not return JSON content"):
        asyncio.run(_client().complete_json(model="m", system="s", user="u", schema=_Schema))


def test_think_flag_is_included_when_set(mock_transport) -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    mock_transport(handler)
    asyncio.run(_client(think=True).complete(model="m", system="s", user="u"))

    assert captured["payload"]["think"] is True


def test_http_status_error_is_not_retried(mock_transport) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, json={"error": "bad request"})

    mock_transport(handler)
    with pytest.raises(LLMError, match="Ollama request failed"):
        asyncio.run(_client().complete(model="m", system="s", user="u"))

    assert calls["count"] == 1


def test_transient_timeout_is_retried_then_succeeds(mock_transport) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    mock_transport(handler)
    result = asyncio.run(_client(retry_backoff_seconds=0).complete(model="m", system="s", user="u"))

    assert result == "ok"
    assert calls["count"] == 2


def test_transient_timeout_raises_after_max_retries(mock_transport) -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        raise httpx.ConnectError("refused", request=request)

    mock_transport(handler)
    with pytest.raises(LLMError, match="failed after 2 attempts"):
        asyncio.run(_client(retry_backoff_seconds=0).complete(model="m", system="s", user="u"))

    assert calls["count"] == 2


def test_non_dict_response_raises(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, json=[1, 2, 3]))

    with pytest.raises(LLMError, match="unexpected response"):
        asyncio.run(_client().complete(model="m", system="s", user="u"))


def test_static_llm_returns_queued_responses_in_order() -> None:
    llm = StaticLLM(['{"ok": true}', "plain text"])

    first = asyncio.run(llm.complete_json(model="m", system="s", user="u", schema=_Schema))
    second = asyncio.run(llm.complete(model="m", system="s", user="u"))

    assert first.ok is True
    assert second == "plain text"


def test_static_llm_raises_when_exhausted() -> None:
    llm = StaticLLM([])

    with pytest.raises(LLMError, match="no remaining response"):
        asyncio.run(llm.complete(model="m", system="s", user="u"))
