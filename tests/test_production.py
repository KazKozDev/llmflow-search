import asyncio

import httpx
import pytest

from deep_research.config import AppConfig, ModelsConfig, OllamaConfig, SearchConfig
from deep_research.models import SearchResult, Source
from deep_research.production import run_production_smoke
from deep_research.tools import ToolError


def _config(**search_overrides) -> AppConfig:
    return AppConfig(
        models=ModelsConfig(
            analyzer="test-model",
            planner="test-model",
            researcher="test-model",
            fact_checker="test-model",
            writer="test-model",
        ),
        ollama=OllamaConfig(base_url="http://ollama.test", timeout_seconds=5, max_retries=1),
        search=SearchConfig(base_url="http://searxng.test", **search_overrides),
    )


def _patch_config(monkeypatch, config: AppConfig) -> None:
    monkeypatch.setattr("deep_research.production.load_config", lambda path: config)


def _checks_by_name(result: dict) -> dict:
    return {check["name"]: check for check in result["checks"]}


class _StubFootnote:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return [SearchResult(title="Footnote article", url="https://footnote.test/a")]

    async def fetch(self, result: SearchResult) -> Source:
        return Source(
            url=result.url,
            canonical_url=result.url,
            title="Footnote article",
            content_hash="sha256:x",
            quality_score=0.6,
            text="footnote text " * 20,
        )


class _FailingStubFootnote(_StubFootnote):
    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        raise ToolError("footnote-mcp unavailable")


def _success_handler(request: httpx.Request) -> httpx.Response:
    host, path = request.url.host, request.url.path
    if host == "ollama.test" and path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "test-model"}]})
    if host == "ollama.test" and path == "/api/chat":
        return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})
    if host == "searxng.test" and path == "/search":
        return httpx.Response(
            200, json={"results": [{"title": "Article", "url": "https://article.test/page", "content": "..."}]}
        )
    if host == "article.test":
        body = "<html><title>Article</title><body>" + ("word " * 40) + "</body></html>"
        return httpx.Response(200, headers={"content-type": "text/html"}, text=body)
    raise AssertionError(f"unexpected request to {request.url}")


@pytest.fixture(autouse=True)
def _no_dns_lookups(monkeypatch):
    async def _allow(host: str) -> None:
        return None

    monkeypatch.setattr("deep_research.tools._ensure_public_host", _allow)


def test_full_success_path(tmp_path, mock_transport, monkeypatch) -> None:
    _patch_config(monkeypatch, _config(provider="searxng"))
    mock_transport(_success_handler)

    result = asyncio.run(run_production_smoke("unused.yaml", tmp_path / "smoke.sqlite3"))

    checks = _checks_by_name(result)
    assert result["passed"] is True
    assert checks["ollama_models"]["passed"] is True
    assert checks["ollama_structured_inference"]["passed"] is True
    assert checks["searxng_search"]["passed"] is True
    assert checks["sqlite_write"]["passed"] is True


def test_ollama_unreachable_skips_inference_and_reports_unknown_provider(tmp_path, mock_transport, monkeypatch) -> None:
    _patch_config(monkeypatch, _config(provider="not-a-real-provider"))

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    mock_transport(handler)
    result = asyncio.run(run_production_smoke("unused.yaml", tmp_path / "smoke.sqlite3"))

    checks = _checks_by_name(result)
    assert result["passed"] is False
    assert checks["ollama_models"]["passed"] is False
    assert checks["ollama_structured_inference"]["passed"] is False
    assert "configured models are unavailable" in checks["ollama_structured_inference"]["detail"]
    assert checks["searxng_search"]["passed"] is False
    assert "unknown search provider" in checks["searxng_search"]["detail"]


def test_models_available_but_inference_fails(tmp_path, mock_transport, monkeypatch) -> None:
    _patch_config(monkeypatch, _config(provider="footnote_mcp"))
    monkeypatch.setattr("deep_research.production.FootnoteMCPProvider", _StubFootnote)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(500, json={"error": "model crashed"})
        raise AssertionError(f"unexpected request to {request.url}")

    mock_transport(handler)
    result = asyncio.run(run_production_smoke("unused.yaml", tmp_path / "smoke.sqlite3"))

    checks = _checks_by_name(result)
    assert checks["ollama_models"]["passed"] is True
    assert checks["ollama_structured_inference"]["passed"] is False
    assert checks["searxng_search"]["passed"] is True


def test_footnote_mcp_primary_provider_failure(tmp_path, mock_transport, monkeypatch) -> None:
    _patch_config(monkeypatch, _config(provider="footnote_mcp"))
    monkeypatch.setattr("deep_research.production.FootnoteMCPProvider", _FailingStubFootnote)
    mock_transport(lambda request: httpx.Response(500))

    result = asyncio.run(run_production_smoke("unused.yaml", tmp_path / "smoke.sqlite3"))

    checks = _checks_by_name(result)
    assert checks["searxng_search"]["passed"] is False
    assert "footnote-mcp unavailable" in checks["searxng_search"]["detail"]


def test_searxng_failure_without_fallback(tmp_path, mock_transport, monkeypatch) -> None:
    _patch_config(monkeypatch, _config(provider="searxng", fallback_provider=None))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})
        if request.url.path == "/search":
            return httpx.Response(500)
        raise AssertionError(f"unexpected request to {request.url}")

    mock_transport(handler)
    result = asyncio.run(run_production_smoke("unused.yaml", tmp_path / "smoke.sqlite3"))

    assert _checks_by_name(result)["searxng_search"]["passed"] is False


def test_searxng_failure_falls_back_to_footnote_mcp_success(tmp_path, mock_transport, monkeypatch) -> None:
    _patch_config(monkeypatch, _config(provider="searxng", fallback_provider="footnote_mcp"))
    monkeypatch.setattr("deep_research.production.FootnoteMCPProvider", _StubFootnote)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})
        if request.url.path == "/search":
            return httpx.Response(500)
        raise AssertionError(f"unexpected request to {request.url}")

    mock_transport(handler)
    result = asyncio.run(run_production_smoke("unused.yaml", tmp_path / "smoke.sqlite3"))

    checks = _checks_by_name(result)
    assert checks["searxng_search"]["passed"] is True
    assert "footnote-mcp fallback works" in checks["searxng_search"]["detail"]


def test_searxng_failure_and_footnote_mcp_fallback_both_fail(tmp_path, mock_transport, monkeypatch) -> None:
    _patch_config(monkeypatch, _config(provider="searxng", fallback_provider="footnote_mcp"))
    monkeypatch.setattr("deep_research.production.FootnoteMCPProvider", _FailingStubFootnote)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "test-model"}]})
        if request.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": '{"ok": true}'}})
        if request.url.path == "/search":
            return httpx.Response(500)
        raise AssertionError(f"unexpected request to {request.url}")

    mock_transport(handler)
    result = asyncio.run(run_production_smoke("unused.yaml", tmp_path / "smoke.sqlite3"))

    checks = _checks_by_name(result)
    assert checks["searxng_search"]["passed"] is False
    assert "footnote-mcp fallback also failed" in checks["searxng_search"]["detail"]


def test_sqlite_write_failure_when_database_path_is_a_directory(tmp_path, mock_transport, monkeypatch) -> None:
    _patch_config(monkeypatch, _config(provider="not-a-real-provider"))
    mock_transport(lambda request: httpx.Response(500))

    result = asyncio.run(run_production_smoke("unused.yaml", tmp_path))

    assert _checks_by_name(result)["sqlite_write"]["passed"] is False


def test_repeated_smoke_run_reuses_existing_smoke_record(tmp_path, mock_transport, monkeypatch) -> None:
    """The smoke run always writes to research_id 'smoke'; a second run against the same
    database hits the duplicate-primary-key path, which is swallowed so the check still
    passes rather than failing the whole smoke check on a rerun."""
    _patch_config(monkeypatch, _config(provider="not-a-real-provider"))
    mock_transport(lambda request: httpx.Response(500))
    database_path = tmp_path / "smoke.sqlite3"

    asyncio.run(run_production_smoke("unused.yaml", database_path))
    result = asyncio.run(run_production_smoke("unused.yaml", database_path))

    assert _checks_by_name(result)["sqlite_write"]["passed"] is True
