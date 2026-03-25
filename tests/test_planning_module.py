"""Unit tests for search plan revision behavior."""

import unittest
from unittest.mock import MagicMock

from core.planning_module import PlanningModule


class PlanningModuleTests(unittest.TestCase):
    """Verify plan creation and revision behavior."""

    def test_revise_plan_uses_duckduckgo_results_from_agent_memory(self):
        """Revisions should recognize the source format emitted by AgentCore."""
        llm_service = MagicMock()
        llm_service.generate_response.return_value = (
            '{"follow_up_searches": ["follow up topic"]}'
        )
        planning = PlanningModule(llm_service)
        plan = {
            "steps": [
                {
                    "type": "search_duckduckgo",
                    "query": "initial query",
                    "description": "Initial search",
                }
            ]
        }
        memory = [
            {
                "type": "search_results",
                "source": "search_duckduckgo",
                "query": "initial query",
                "results": [
                    {
                        "title": "Result A",
                        "snippet": "Useful context",
                    }
                ],
            }
        ]

        revised_plan = planning.revise_plan(
            plan,
            memory,
            plan["steps"][0],
        )

        self.assertEqual(len(revised_plan["steps"]), 2)
        self.assertEqual(revised_plan["steps"][1]["query"], "follow up topic")


if __name__ == "__main__":
    unittest.main()