import asyncio
import hashlib
import socket

import httpx
import pytest

from deep_research.llm import LLMError
from deep_research.models import SearchResult, Source
from deep_research.store import EvidenceStore
from deep_research.tools import (
    ClassifyingFetcher,
    DomainClassifier,
    DomainVerdict,
    FallbackFetcher,
    FallbackSearchProvider,
    FootnoteMCPProvider,
    PageFetcher,
    SearchProvider,
    SearxNGProvider,
    ToolError,
    _ensure_public_host,
    classify_source_type,
)


def test_classifies_primary_and_reputable_news_domains() -> None:
    assert classify_source_type("https://openai.com/news/model-release") == "official_documentation"
    assert classify_source_type("https://arxiv.org/abs/2607.12345") == "institutional"
    assert classify_source_type("https://www.reuters.com/technology/ai/") == "news"
    assert classify_source_type("https://example.org/blog") == "web_page"


def test_classifies_government_and_international_domains() -> None:
    assert classify_source_type("https://www.lamoncloa.gob.es/lang/en/") == "official_documentation"
    assert classify_source_type("https://administracion.gob.es/") == "official_documentation"
    assert classify_source_type("https://commission.europa.eu/index_en") == "official_documentation"
    assert classify_source_type("https://ec.europa.eu/economy_finance/") == "official_documentation"
    assert classify_source_type("https://www.gov.uk/government/news") == "official_documentation"
    assert classify_source_type("https://www.economie.gouv.fr/") == "official_documentation"
    assert classify_source_type("https://www.nasa.gov/missions") == "official_documentation"
    assert classify_source_type("https://www.who.int/news") == "institutional"
    assert classify_source_type("https://www.euronews.com/business") == "news"
    assert classify_source_type("https://elpais.com/economia/") == "news"
    # Not government: "gov"-like fragments inside ordinary names must not match.
    assert classify_source_type("https://governance-blog.com/post") == "web_page"


class _StubProvider(SearchProvider):
    def __init__(self, results: list[SearchResult] | None = None, error: bool = False) -> None:
        self.results = results or []
        self.error = error
        self.calls = 0

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.calls += 1
        if self.error:
            raise ToolError("stub provider failed")
        return self.results


class _StubFetcher:
    def __init__(self, error: bool = False) -> None:
        self.error = error
        self.calls = 0

    async def fetch(self, result: SearchResult) -> Source:
        self.calls += 1
        if self.error:
            raise ToolError("stub fetcher failed")
        text = "fetched page text"
        return Source(
            url=result.url,
            canonical_url=result.url,
            title=result.title,
            content_hash=f"sha256:{hashlib.sha256(text.encode()).hexdigest()}",
            quality_score=0.5,
            text=text,
        )


def _result() -> SearchResult:
    return SearchResult(title="Example", url="https://example.org/page")


def test_fallback_search_uses_primary_when_it_returns_results() -> None:
    primary = _StubProvider(results=[_result()])
    fallback = _StubProvider(results=[_result()])
    provider = FallbackSearchProvider(primary, fallback)

    results = asyncio.run(provider.search("query", 5))

    assert len(results) == 1
    assert fallback.calls == 0


def test_fallback_search_kicks_in_on_empty_results() -> None:
    primary = _StubProvider(results=[])
    fallback = _StubProvider(results=[_result()])
    provider = FallbackSearchProvider(primary, fallback)

    results = asyncio.run(provider.search("query", 5))

    assert len(results) == 1
    assert primary.calls == 1
    assert fallback.calls == 1


def test_fallback_search_kicks_in_on_primary_error() -> None:
    primary = _StubProvider(error=True)
    fallback = _StubProvider(results=[_result()])
    provider = FallbackSearchProvider(primary, fallback)

    results = asyncio.run(provider.search("query", 5))

    assert len(results) == 1
    assert fallback.calls == 1


def test_fallback_fetcher_uses_primary_on_success() -> None:
    primary = _StubFetcher()
    fallback = _StubFetcher()
    fetcher = FallbackFetcher(primary, fallback)

    source = asyncio.run(fetcher.fetch(_result()))

    assert source.text == "fetched page text"
    assert fallback.calls == 0


def test_fallback_fetcher_kicks_in_on_primary_tool_error() -> None:
    primary = _StubFetcher(error=True)
    fallback = _StubFetcher()
    fetcher = FallbackFetcher(primary, fallback)

    source = asyncio.run(fetcher.fetch(_result()))

    assert source.text == "fetched page text"
    assert primary.calls == 1
    assert fallback.calls == 1


class _ClassifierLLM:
    def __init__(self, source_type: str = "news", confidence: float = 0.9, error: bool = False) -> None:
        self.source_type = source_type
        self.confidence = confidence
        self.error = error
        self.calls = 0

    async def complete_json(self, *, model: str, system: str, user: str, schema):
        self.calls += 1
        if self.error:
            raise LLMError("classifier model unavailable")
        return DomainVerdict(source_type=self.source_type, confidence=self.confidence)

    async def complete(self, *, model: str, system: str, user: str) -> str:
        raise AssertionError("not used")


def _classifier(tmp_path, llm) -> tuple[DomainClassifier, EvidenceStore]:
    store = EvidenceStore(tmp_path / "cache.sqlite3")
    return DomainClassifier(llm, "test-model", store), store


def test_domain_classifier_caches_one_llm_call_per_domain(tmp_path) -> None:
    llm = _ClassifierLLM(source_type="news", confidence=0.9)
    classifier, store = _classifier(tmp_path, llm)
    try:
        first = asyncio.run(classifier.classify("https://unknown-outlet.example/article/1", "Big story"))
        second = asyncio.run(classifier.classify("https://unknown-outlet.example/other/2", "Other story"))

        assert first == ("news", 0.7)
        assert second == first
        assert llm.calls == 1
    finally:
        store.close()


def test_domain_classifier_caps_llm_trust_below_curated_tiers(tmp_path) -> None:
    llm = _ClassifierLLM(source_type="institutional", confidence=0.95)
    classifier, store = _classifier(tmp_path, llm)
    try:
        verdict = asyncio.run(classifier.classify("https://some-institute.example/", "Institute"))
        assert verdict is not None
        assert verdict[1] == DomainClassifier.MAX_LLM_QUALITY
    finally:
        store.close()


def test_domain_classifier_low_confidence_degrades_to_web_page(tmp_path) -> None:
    llm = _ClassifierLLM(source_type="news", confidence=0.3)
    classifier, store = _classifier(tmp_path, llm)
    try:
        verdict = asyncio.run(classifier.classify("https://maybe-news.example/", "Story"))
        assert verdict == ("web_page", 0.55)
    finally:
        store.close()


def test_domain_classifier_returns_none_on_llm_failure_without_caching(tmp_path) -> None:
    llm = _ClassifierLLM(error=True)
    classifier, store = _classifier(tmp_path, llm)
    try:
        assert asyncio.run(classifier.classify("https://flaky.example/", "Page")) is None
        assert store.get_domain_classification("flaky.example") is None
    finally:
        store.close()


class _KnownQualityFetcher:
    def __init__(self, url: str, quality: float = 0.55) -> None:
        self.url = url
        self.quality = quality

    async def fetch(self, result: SearchResult) -> Source:
        return Source(
            url=self.url,
            canonical_url=self.url,
            title="Page",
            content_hash="sha256:x",
            quality_score=self.quality,
            source_type="web_page",
            text="text",
        )


def test_classifying_fetcher_upgrades_unknown_domains(tmp_path) -> None:
    llm = _ClassifierLLM(source_type="news", confidence=0.9)
    classifier, store = _classifier(tmp_path, llm)
    try:
        fetcher = ClassifyingFetcher(_KnownQualityFetcher("https://unknown-outlet.example/a"), classifier)
        source = asyncio.run(fetcher.fetch(_result()))
        assert source.source_type == "news"
        assert source.quality_score == 0.7
    finally:
        store.close()


def test_classifying_fetcher_skips_domains_the_static_rules_know(tmp_path) -> None:
    llm = _ClassifierLLM()
    classifier, store = _classifier(tmp_path, llm)
    try:
        fetcher = ClassifyingFetcher(_KnownQualityFetcher("https://www.reuters.com/a", quality=0.7), classifier)
        source = asyncio.run(fetcher.fetch(_result()))
        assert llm.calls == 0
        assert source.quality_score == 0.7
    finally:
        store.close()


def test_readable_text_guard_rejects_raw_pdf_and_binary() -> None:
    from deep_research.tools import looks_like_readable_text

    assert looks_like_readable_text("Обычный текст страницы о стратегии ИИ. " * 10)
    assert not looks_like_readable_text("%PDF-1.3\n%��� stream x�+TT")
    assert not looks_like_readable_text("data " + "�" * 500)


class _StubbedMCPProvider:
    """FootnoteMCPProvider with _call_tool replaced by canned responses."""

    def __new__(cls, responses):
        from deep_research.tools import FootnoteMCPProvider

        provider = FootnoteMCPProvider("/bin/true", [], "ru", "auto", False, 0)
        provider._calls = []

        async def fake_call_tool(tool_name, arguments):
            provider._calls.append(tool_name)
            return responses[tool_name]

        provider._call_tool = fake_call_tool
        return provider


def test_mcp_fetch_routes_pdf_urls_to_file_parser() -> None:
    provider = _StubbedMCPProvider(
        {"web_parse_file": {"pages": [{"page": 1, "text": "Стратегия развития ИИ до 2030 года. " * 5}]}}
    )
    result = SearchResult(title="Strategy", url="https://static.kremlin.ru/files/strategy.PDF")

    source = asyncio.run(provider.fetch(result))

    assert provider._calls == ["web_parse_file"]
    assert "Стратегия" in source.text
    assert source.quality_score == 0.82  # kremlin.ru stays official via domain rules


def test_mcp_fetch_falls_back_to_file_parser_on_binary_web_read() -> None:
    provider = _StubbedMCPProvider(
        {
            "web_read": {"text": "%PDF-1.3 " + "�" * 300},
            "web_parse_file": {"pages": [{"page": 1, "text": "Readable strategy text from the parsed document. " * 4}]},
        }
    )
    result = SearchResult(title="Doc", url="https://example.org/document")

    source = asyncio.run(provider.fetch(result))

    assert provider._calls == ["web_read", "web_parse_file"]
    assert "Readable strategy text" in source.text


def test_mcp_fetch_fails_when_file_parser_returns_nothing() -> None:
    import pytest

    provider = _StubbedMCPProvider({"web_parse_file": {"pages": []}})
    result = SearchResult(title="Scan", url="https://example.org/scan.pdf")

    with pytest.raises(ToolError, match="could not extract readable text"):
        asyncio.run(provider.fetch(result))


def test_mcp_search_normalizes_results() -> None:
    provider = _StubbedMCPProvider(
        {
            "web_search": {
                "results": [
                    {"title": "A", "url": "https://example.org/a", "snippet": "snip"},
                    {"url": "https://example.org/b"},
                    {"no_url": True},
                ]
            }
        }
    )

    results = asyncio.run(provider.search("query", 2))

    assert len(results) == 2
    assert results[0].provider == "footnote_mcp"
    assert results[0].snippet == "snip"


def test_mcp_search_raises_on_invalid_results_type() -> None:
    provider = _StubbedMCPProvider({"web_search": {"results": "not-a-list"}})

    with pytest.raises(ToolError, match="invalid web_search result"):
        asyncio.run(provider.search("query", 2))


def test_mcp_fetch_rejects_non_dict_web_read_payload() -> None:
    provider = _StubbedMCPProvider({"web_read": "not-a-dict"})
    result = SearchResult(title="Doc", url="https://example.org/a")

    with pytest.raises(ToolError, match="invalid web_read result"):
        asyncio.run(provider.fetch(result))


def test_mcp_fetch_rejects_too_short_text() -> None:
    provider = _StubbedMCPProvider({"web_read": {"text": "short"}})
    result = SearchResult(title="Doc", url="https://example.org/a")

    with pytest.raises(ToolError, match="did not return enough readable page text"):
        asyncio.run(provider.fetch(result))


def test_mcp_fetch_full_web_read_success_with_metadata() -> None:
    provider = _StubbedMCPProvider(
        {
            "web_read": {
                "text": "Readable page content. " * 10,
                "url": "https://example.org/canonical",
                "title": "Real Title",
                "author": "Jane Doe",
                "published_at": "2026-01-01",
                "source_type": {"source_type": "news"},
                "quality_score": 0.9,
            }
        }
    )
    result = SearchResult(title="Fallback", url="https://example.org/a")

    source = asyncio.run(provider.fetch(result))

    assert source.title == "Real Title"
    assert source.author == "Jane Doe"
    assert source.source_type == "news"
    assert source.quality_score == 0.9


def test_mcp_fetch_uses_string_source_type_and_domain_quality_floor() -> None:
    provider = _StubbedMCPProvider(
        {"web_read": {"text": "Readable page content. " * 10, "source_type": "official_documentation"}}
    )
    result = SearchResult(title="Doc", url="https://openai.com/blog/post")

    source = asyncio.run(provider.fetch(result))

    assert source.source_type == "official_documentation"
    assert source.quality_score == 0.82


def test_mcp_fetch_parsed_file_rejects_non_dict_payload() -> None:
    provider = _StubbedMCPProvider({"web_parse_file": "not-a-dict"})
    result = SearchResult(title="Scan", url="https://example.org/scan.pdf")

    with pytest.raises(ToolError, match="invalid web_parse_file result"):
        asyncio.run(provider.fetch(result))


def test_mcp_call_tool_rejects_missing_executable() -> None:
    provider = FootnoteMCPProvider("/no/such/footnote-mcp-binary", [], "en", "auto", False, 0)
    result = SearchResult(title="Doc", url="https://example.org/a")

    with pytest.raises(ToolError, match="footnote-mcp executable is missing"):
        asyncio.run(provider.fetch(result))


def test_searxng_search_parses_results(mock_transport) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["q"] == "local RAG"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"title": "First", "url": "https://example.org/1", "content": "snippet one"},
                    {"title": "Second", "url": "https://example.org/2", "publishedDate": "2026-01-01"},
                    {"url": "https://example.org/3"},
                    {"no_url": True},
                ]
            },
        )

    mock_transport(handler)
    provider = SearxNGProvider("http://searxng.test", "en")

    results = asyncio.run(provider.search("local RAG", 4))

    assert len(results) == 3
    assert results[0].title == "First"
    assert results[0].snippet == "snippet one"
    assert results[1].published_at == "2026-01-01"
    assert results[2].title == "https://example.org/3"


def test_searxng_search_raises_tool_error_on_http_failure(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(500))
    provider = SearxNGProvider("http://searxng.test", "en")

    with pytest.raises(ToolError, match="SearXNG search failed"):
        asyncio.run(provider.search("query", 5))


def test_searxng_search_raises_tool_error_on_invalid_json(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, text="not json"))
    provider = SearxNGProvider("http://searxng.test", "en")

    with pytest.raises(ToolError, match="SearXNG search failed"):
        asyncio.run(provider.search("query", 5))


def test_searxng_search_ignores_non_dict_payload(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, json=["not", "a", "dict"]))
    provider = SearxNGProvider("http://searxng.test", "en")

    results = asyncio.run(provider.search("query", 5))

    assert results == []


@pytest.fixture(autouse=True)
def _no_real_dns(monkeypatch):
    async def _allow(host: str) -> None:
        return None

    monkeypatch.setattr("deep_research.tools._ensure_public_host", _allow)


def test_page_fetcher_extracts_readable_text_and_title(mock_transport) -> None:
    body = (
        "<html><head><title>My Article</title><script>ignored()</script></head>\n  \n"
        "<body><nav>skip nav</nav><h1>Heading</h1><p>" + ("Body text. " * 15) + "</p></body></html>"
    )
    mock_transport(lambda request: httpx.Response(200, headers={"content-type": "text/html"}, text=body))
    fetcher = PageFetcher()

    source = asyncio.run(fetcher.fetch(SearchResult(title="Fallback title", url="https://example.org/article")))

    assert source.title == "My Article"
    assert "Body text." in source.text
    assert "skip nav" not in source.text
    assert "ignored()" not in source.text


def test_page_fetcher_rejects_non_http_scheme() -> None:
    fetcher = PageFetcher()

    with pytest.raises(ToolError, match="Only public HTTP"):
        asyncio.run(fetcher.fetch(SearchResult(title="Local", url="file:///etc/passwd")))


def test_page_fetcher_rejects_unsupported_content_type(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF"))
    fetcher = PageFetcher()

    with pytest.raises(ToolError, match="Unsupported content type"):
        asyncio.run(fetcher.fetch(SearchResult(title="Doc", url="https://example.org/file.pdf")))


def test_page_fetcher_rejects_oversized_page(mock_transport) -> None:
    huge_body = "<html><body>" + ("x" * 200) + "</body></html>"
    mock_transport(lambda request: httpx.Response(200, headers={"content-type": "text/html"}, text=huge_body))
    fetcher = PageFetcher(max_bytes=50)

    with pytest.raises(ToolError, match="exceeds configured size limit"):
        asyncio.run(fetcher.fetch(SearchResult(title="Big", url="https://example.org/big")))


def test_page_fetcher_rejects_too_short_text(mock_transport) -> None:
    mock_transport(lambda request: httpx.Response(200, headers={"content-type": "text/html"}, text="<p>short</p>"))
    fetcher = PageFetcher()

    with pytest.raises(ToolError, match="did not contain enough readable text"):
        asyncio.run(fetcher.fetch(SearchResult(title="Short", url="https://example.org/short")))


def test_page_fetcher_wraps_http_errors(mock_transport) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    mock_transport(handler)
    fetcher = PageFetcher()

    with pytest.raises(ToolError, match="Page fetch failed"):
        asyncio.run(fetcher.fetch(SearchResult(title="Down", url="https://example.org/down")))


def test_ensure_public_host_rejects_localhost() -> None:
    with pytest.raises(ToolError, match="local address"):
        asyncio.run(_ensure_public_host("localhost"))


def test_ensure_public_host_rejects_unresolvable_host(monkeypatch) -> None:
    def fail_getaddrinfo(*args, **kwargs):
        raise socket.gaierror("name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fail_getaddrinfo)

    with pytest.raises(ToolError, match="Cannot resolve host"):
        asyncio.run(_ensure_public_host("no-such-host.invalid"))


def test_ensure_public_host_rejects_private_address(monkeypatch) -> None:
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    with pytest.raises(ToolError, match="non-public address"):
        asyncio.run(_ensure_public_host("internal.example"))


def test_ensure_public_host_allows_public_address(monkeypatch) -> None:
    def fake_getaddrinfo(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    asyncio.run(_ensure_public_host("example.org"))


def test_search_provider_base_class_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        asyncio.run(SearchProvider().search("query", 1))


def test_readable_text_guard_rejects_empty_sample() -> None:
    from deep_research.tools import looks_like_readable_text

    assert not looks_like_readable_text("")


def test_domain_classifier_returns_none_for_url_without_hostname(tmp_path) -> None:
    llm = _ClassifierLLM()
    classifier, store = _classifier(tmp_path, llm)
    try:
        assert asyncio.run(classifier.classify("not-a-url", "Title")) is None
        assert llm.calls == 0
    finally:
        store.close()


def test_classify_source_type_repository_domain() -> None:
    assert classify_source_type("https://github.com/org/repo") == "repository"


def test_classify_source_type_docs_subdomain_prefix() -> None:
    assert classify_source_type("https://docs.example.com/guide") == "official_documentation"
