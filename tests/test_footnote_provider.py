import pytest

from deep_research.tools import FootnoteMCPProvider


class StubFootnoteProvider(FootnoteMCPProvider):
    def __init__(self, responses: dict[str, object]) -> None:
        super().__init__("/missing/footnote-mcp", [], "en", "auto", False)
        self.responses = responses

    async def _call_tool(self, tool_name: str, arguments: dict) -> object:
        return self.responses[tool_name]


@pytest.mark.asyncio
async def test_normalizes_footnote_search_and_read_response() -> None:
    provider = StubFootnoteProvider(
        {
            "web_search": {
                "results": [
                    {
                        "title": "Primary source",
                        "url": "https://example.org/source",
                        "snippet": "A concise snippet",
                    }
                ]
            },
            "web_read": {
                "url": "https://example.org/source",
                "title": "Primary source",
                "source_type": {"source_type": "aggregator", "reasons": []},
                "text": "A" * 150,
            },
        }
    )

    results = await provider.search("example", 5)
    source = await provider.fetch(results[0])

    assert results[0].provider == "footnote_mcp"
    assert source.source_type == "aggregator"
    assert source.quality_score == 0.55


@pytest.mark.asyncio
async def test_rejects_invalid_footnote_search_payload() -> None:
    provider = StubFootnoteProvider({"web_search": {"results": "not a list"}})

    with pytest.raises(Exception, match="invalid web_search"):
        await provider.search("example", 5)
