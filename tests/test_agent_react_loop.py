"""Unit tests for the agent execution loop."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from core.agent_core import AgentCore
from core.planning_module import PlanningModule


class FakeMemory:
    """Minimal memory implementation for agent tests."""

    def __init__(self):
        self.short_term = []
        self.long_term = []
        self.links = {}

    def add_to_short_term(self, item):
        self.short_term.append(item)

    def get_short_term(self):
        return list(self.short_term)

    def add_to_long_term(self, item):
        self.long_term.append(item)

    def get_long_term(self):
        return list(self.long_term)

    def add_to_links(self, url, title):
        self.links[url] = title

    def get_links(self):
        return dict(self.links)

    def get_relevant_content(self, _query, max_items=100):
        return self.short_term[-max_items:]


class AgentReActLoopTests(unittest.TestCase):
    """Verify the plan, act, reflect loop in AgentCore."""

    def setUp(self):
        self.memory = FakeMemory()
        self.mock_llm = MagicMock()
        self.mock_tools = MagicMock()
        self.mock_planning = MagicMock()
        self.mock_report = MagicMock()

        self.agent = AgentCore(
            memory=self.memory,
            planning=self.mock_planning,
            tools=self.mock_tools,
            report_generator=self.mock_report,
            llm_service=self.mock_llm,
            max_iterations=3,
        )

    def test_react_loop_executes_all_steps(self):
        """The agent should execute planned steps and then report."""
        initial_plan = {
            "steps": [
                {
                    "type": "search_duckduckgo",
                    "query": "step 1 query",
                    "description": "First step",
                },
                {
                    "type": "search_wikipedia",
                    "query": "step 2 query",
                    "description": "Second step",
                },
            ]
        }
        self.mock_planning.create_plan.return_value = initial_plan
        self.mock_planning.revise_plan.return_value = initial_plan
        self.mock_tools.execute_tool = AsyncMock(
            return_value=[
                {
                    "title": "Result",
                    "url": "http://test.com",
                    "snippet": "content",
                }
            ]
        )
        self.mock_tools.parse_top_results = 1
        self.agent._process_search_result = AsyncMock()
        self.mock_report.generate_report.return_value = "# Final Report"

        report = asyncio.run(self.agent.process_query("Test query"))

        self.assertEqual(report, "# Final Report")
        self.mock_planning.create_plan.assert_called_once_with("Test query")
        self.assertEqual(self.mock_tools.execute_tool.call_count, 2)
        self.mock_tools.execute_tool.assert_any_call(
            "search_duckduckgo", query="step 1 query"
        )
        self.mock_tools.execute_tool.assert_any_call(
            "search_wikipedia", query="step 2 query"
        )
        self.assertEqual(self.mock_planning.revise_plan.call_count, 2)

        search_entries = [
            item
            for item in self.memory.get_short_term()
            if item.get("type") == "search_results"
        ]
        self.assertEqual(len(search_entries), 2)
        self.assertEqual(len(self.memory.get_long_term()), 1)

    def test_dynamic_plan_revision_adds_new_step(self):
        """The agent should execute steps added by revise_plan."""
        initial_plan = {
            "steps": [
                {
                    "type": "search_duckduckgo",
                    "query": "initial query",
                    "description": "Step 1",
                }
            ]
        }
        revised_plan = {
            "steps": [
                {
                    "type": "search_duckduckgo",
                    "query": "initial query",
                    "description": "Step 1",
                },
                {
                    "type": "search_arxiv",
                    "query": "new query",
                    "description": "Step 2",
                },
            ]
        }

        self.mock_planning.create_plan.return_value = initial_plan
        self.mock_planning.revise_plan.side_effect = [
            revised_plan,
            revised_plan,
        ]
        self.mock_tools.execute_tool = AsyncMock(return_value=[])
        self.mock_tools.parse_top_results = 1
        self.mock_report.generate_report.return_value = "Report"

        asyncio.run(self.agent.process_query("Dynamic test"))

        self.assertEqual(self.mock_tools.execute_tool.call_count, 2)
        self.mock_tools.execute_tool.assert_any_call(
            "search_arxiv", query="new query"
        )

    def test_process_query_executes_follow_up_step_from_real_revision(self):
        """The real planner should add and execute a follow-up DuckDuckGo step."""
        llm_service = MagicMock()
        llm_service.generate_response.side_effect = [
            (
                '{"main_keywords": ["initial query"], '
                '"wikipedia_topics": [], '
                '"alternative_keywords": [], '
                '"subtopics": []}'
            ),
            '{"follow_up_searches": ["follow up topic"]}',
        ]
        llm_service.generate_response_async = AsyncMock(return_value="NO")
        planning = PlanningModule(llm_service)
        tools = MagicMock()
        tools.parse_top_results = 1
        tools.execute_tool = AsyncMock(
            side_effect=[
                [
                    {
                        "title": "Result",
                        "url": "http://test.com",
                        "snippet": "content",
                    }
                ],
                [
                    {
                        "title": "Follow up result",
                        "url": "http://test-2.com",
                        "snippet": "more content",
                    }
                ],
            ]
        )
        report_generator = MagicMock()
        report_generator.generate_report.return_value = "# Final Report"
        agent = AgentCore(
            memory=FakeMemory(),
            planning=planning,
            tools=tools,
            report_generator=report_generator,
            llm_service=llm_service,
            max_iterations=3,
        )
        agent._process_search_result = AsyncMock()

        report = asyncio.run(agent.process_query("initial query"))

        self.assertEqual(report, "# Final Report")
        self.assertEqual(tools.execute_tool.call_count, 2)
        tools.execute_tool.assert_any_call(
            "search_duckduckgo", query="initial query"
        )
        tools.execute_tool.assert_any_call(
            "search_duckduckgo", query="follow up topic"
        )


if __name__ == "__main__":
    unittest.main()
