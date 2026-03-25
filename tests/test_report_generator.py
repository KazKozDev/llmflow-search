"""Regression tests for report generation."""

import unittest
from unittest.mock import MagicMock

from core.report_generator import ReportGenerator


class FakeMemory:
    """Minimal memory stub for report generator tests."""

    def __init__(self, relevant_items, links):
        self._relevant_items = relevant_items
        self._links = links

    def get_relevant_content(self, _query, max_items=20):
        return self._relevant_items[:max_items]

    def get_links(self):
        return self._links


class ReportGeneratorTests(unittest.TestCase):
    """Verify report generator uses collected search results."""

    def test_search_results_snippets_are_included_in_prompt(self):
        """DuckDuckGo snippets should feed report context.

        This should work even when no pages were parsed in depth.
        """
        memory = FakeMemory(
            relevant_items=[
                {
                    "type": "search_results",
                    "query": "новости ллм 2026",
                    "results": [
                        {
                            "title": "LLM News Today (March 2026)",
                            "url": "https://llm-stats.com/ai-news",
                            "snippet": (
                                "March 2026 saw several notable LLM releases "
                                "and benchmark shifts."
                            ),
                        }
                    ],
                }
            ],
            links={
                "https://llm-stats.com/ai-news": "LLM News Today (March 2026)",
            },
        )
        llm_service = MagicMock()
        llm_service.generate_response.return_value = "Summary paragraph [1]"
        generator = ReportGenerator(memory, llm_service)

        report = generator.generate_report("новости ллм 2026")

        prompt = llm_service.generate_response.call_args[0][0]
        self.assertIn("LLM News Today (March 2026)", prompt)
        self.assertIn(
            "March 2026 saw several notable LLM releases",
            prompt,
        )
        self.assertIn("Available source list", prompt)
        self.assertIn("## References", report)
        self.assertIn("https://llm-stats.com/ai-news", report)


if __name__ == "__main__":
    unittest.main()
