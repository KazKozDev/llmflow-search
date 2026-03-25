"""Unit tests for search-step result handling."""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.agent_core import AgentCore
from core.tools_module import ToolsModule


class FakeMemory:
    """Minimal memory implementation for tool-handling tests."""

    def __init__(self):
        self.short_term = []
        self.links = {}

    def add_to_short_term(self, item):
        self.short_term.append(item)

    def get_short_term(self):
        return list(self.short_term)

    def add_to_links(self, url, title):
        self.links[url] = title

    def get_links(self):
        return dict(self.links)

    def get_relevant_content(self, _query, max_items=100):
        return self.short_term[-max_items:]


class ToolUsageTests(unittest.IsolatedAsyncioTestCase):
    """Verify per-tool result normalization inside AgentCore."""

    def setUp(self):
        self.memory = FakeMemory()
        self.tools = MagicMock()
        self.tools.parse_top_results = 2
        self.tools.parse_duckduckgo_result = AsyncMock(return_value="parsed")
        self.llm_service = MagicMock()
        self.llm_service.generate_response_async = AsyncMock(return_value="NO")

        self.agent = AgentCore(
            memory=self.memory,
            planning=MagicMock(),
            tools=self.tools,
            report_generator=MagicMock(),
            llm_service=self.llm_service,
            max_iterations=2,
        )
        self.agent.current_query = "test query"

    async def test_duckduckgo_results_are_stored_and_linked(self):
        """DuckDuckGo search should store results and add links."""
        results = [
            {
                "title": "Example result",
                "url": "https://example.com",
                "snippet": "Example snippet",
            }
        ]
        self.tools.execute_tool = AsyncMock(return_value=results)

        await self.agent._execute_search_step(
            {
                "type": "search_duckduckgo",
                "query": "example query",
                "description": "Search web",
            },
            {"steps": []},
        )

        self.assertEqual(
            self.memory.links["https://example.com"],
            "Example result",
        )
        self.assertTrue(
            any(
                item.get("type") == "search_results"
                for item in self.memory.short_term
            )
        )
        self.tools.parse_duckduckgo_result.assert_not_awaited()

    async def test_wikipedia_result_uses_page_metadata(self):
        """Wikipedia result should add one formatted reference link."""
        self.tools.execute_tool = AsyncMock(
            return_value={
                "page_found": True,
                "url": "https://en.wikipedia.org/wiki/Test",
                "title": "Test",
            }
        )

        await self.agent._execute_search_step(
            {
                "type": "search_wikipedia",
                "query": "test",
                "description": "Search Wikipedia",
            },
            {"steps": []},
        )

        self.assertEqual(
            self.memory.links["https://en.wikipedia.org/wiki/Test"],
            "Wikipedia: Test",
        )

    async def test_generic_tool_results_add_all_links(self):
        """Generic list results with url and title should be linked."""
        self.tools.execute_tool = AsyncMock(
            return_value=[
                {"title": "Paper A", "url": "https://a.example"},
                {"title": "Paper B", "url": "https://b.example"},
            ]
        )

        await self.agent._execute_search_step(
            {
                "type": "search_searxng",
                "query": "topic",
                "description": "Meta search",
            },
            {"steps": []},
        )

        self.assertEqual(
            self.memory.links["https://a.example"],
            "search_searxng: Paper A",
        )
        self.assertEqual(
            self.memory.links["https://b.example"],
            "search_searxng: Paper B",
        )

    async def test_tool_error_is_recorded_in_memory(self):
        """Tool execution failures should be captured as error items."""
        self.tools.execute_tool = AsyncMock(side_effect=RuntimeError("boom"))

        await self.agent._execute_search_step(
            {
                "type": "search_pubmed",
                "query": "topic",
                "description": "Search PubMed",
            },
            {"steps": []},
        )

        errors = [
            item
            for item in self.memory.short_term
            if item.get("type") == "error"
        ]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["source"], "search_pubmed")
        self.assertIn("boom", errors[0]["error"])

    async def test_wikipedia_urls_use_wikipedia_api_parser(self):
        """Wikipedia URLs should be parsed via the Wikipedia tool, not a missing class."""
        tools_module = ToolsModule(
            memory=FakeMemory(),
            llm_service=MagicMock(),
            config={
                "cache": {
                    "provider": "sqlite",
                    "sqlite_path": ":memory:",
                },
                "rate_limits": {"default": {"requests_per_minute": 30}},
            },
        )
        fake_parsing_cache = MagicMock()
        fake_parsing_cache.get.return_value = None

        with patch(
            "core.parsing_cache.get_parsing_cache",
            return_value=fake_parsing_cache,
        ):
            with patch(
                "core.tools.impl_wikipedia.WikipediaTool.get_article_content",
                new=AsyncMock(return_value={"extract": "Wikipedia content"}),
            ) as get_article_content:
                with patch(
                    "core.tools.async_link_parser.extract_content_from_url_async",
                    new=AsyncMock(return_value="fallback content"),
                ) as fallback_parser:
                    content = await tools_module.parse_duckduckgo_result(
                        {
                            "url": (
                                "https://en.wikipedia.org/wiki/"
                                "Python_(programming_language)"
                            )
                        }
                    )

        self.assertEqual(content, "Wikipedia content")
        get_article_content.assert_awaited_once_with(
            "Python (programming language)",
            language="en",
        )
        fallback_parser.assert_not_awaited()
        fake_parsing_cache.set.assert_called_once_with(
            "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "Wikipedia content",
        )


if __name__ == "__main__":
    unittest.main()
