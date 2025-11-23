"""
Test Agent ReAct Loop
Verifies that the AgentCore correctly implements the Plan -> Act -> Reflect -> Repeat loop.
"""
import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import json

from core.agent_core import AgentCore
from core.memory_module import MemoryModule

class TestAgentReActLoop(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryModule()
        self.mock_llm = MagicMock()
        self.mock_tools = MagicMock()
        self.mock_planning = MagicMock()
        self.mock_report = MagicMock()
        
        # Setup Agent
        self.agent = AgentCore(
            memory=self.memory,
            planning=self.mock_planning,
            tools=self.mock_tools,
            report_generator=self.mock_report,
            llm_service=self.mock_llm,
            max_iterations=3
        )

    def test_react_loop_execution(self):
        """Test that the agent plans, executes, and reflects in a loop."""
        
        # 1. Mock Planning (Plan)
        initial_plan = {
            "steps": [
                {"type": "search_duckduckgo", "query": "step 1 query", "description": "First step"},
                {"type": "search_wikipedia", "query": "step 2 query", "description": "Second step"}
            ]
        }
        self.mock_planning.create_plan.return_value = initial_plan
        
        # 2. Mock Tool Execution (Act)
        # Async mock for execute_tool
        self.mock_tools.execute_tool = AsyncMock(return_value=[{"title": "Result", "url": "http://test.com", "snippet": "content"}])
        self.mock_tools.parse_top_results = 1
        
        # 3. Mock Reflection (Reflect)
        # Simulate a plan revision that doesn't add new steps (so it finishes the original plan)
        # Or we could add a step to test dynamic expansion, but let's keep it simple first.
        self.mock_planning.revise_plan.return_value = initial_plan
        
        # 4. Mock Report Generation
        self.mock_report.generate_report.return_value = "# Final Report"
        
        # Run the agent
        query = "Test query"
        report = asyncio.run(self.agent.process_query(query))
        
        # --- VERIFICATION ---
        
        # 1. Verify Planning was called
        self.mock_planning.create_plan.assert_called_once_with(query)
        print("✓ Planning module called")
        
        # 2. Verify Tools were executed (Act)
        # Should be called twice (once for each step in initial plan)
        self.assertEqual(self.mock_tools.execute_tool.call_count, 2)
        self.mock_tools.execute_tool.assert_any_call("search_duckduckgo", query="step 1 query")
        self.mock_tools.execute_tool.assert_any_call("search_wikipedia", query="step 2 query")
        print(f"✓ Tools executed {self.mock_tools.execute_tool.call_count} times as planned")
        
        # 3. Verify Reflection (Reflect)
        # revise_plan should be called after each step
        self.assertEqual(self.mock_planning.revise_plan.call_count, 2)
        print(f"✓ Reflection (revise_plan) called {self.mock_planning.revise_plan.call_count} times")
        
        # 4. Verify Memory usage
        # Check if steps were recorded in memory
        short_term = self.memory.get_short_term()
        search_entries = [m for m in short_term if m.get("type") == "search_results"]
        self.assertEqual(len(search_entries), 2)
        print("✓ Memory updated with search results")
        
        print("\nSUCCESS: Agent successfully demonstrated Plan -> Act -> Reflect loop!")

    def test_dynamic_plan_update(self):
        """Test that the agent can dynamically add steps during execution (Reflection)."""
        
        # Initial plan has 1 step
        initial_plan = {
            "steps": [
                {"type": "search_duckduckgo", "query": "initial query", "description": "Step 1"}
            ]
        }
        
        # Revised plan adds a NEW step
        revised_plan = {
            "steps": [
                {"type": "search_duckduckgo", "query": "initial query", "description": "Step 1"},
                {"type": "search_arxiv", "query": "new query", "description": "Step 2 (Added dynamically)"}
            ]
        }
        
        self.mock_planning.create_plan.return_value = initial_plan
        
        # revise_plan returns revised_plan on first call, then same plan on subsequent calls
        self.mock_planning.revise_plan.side_effect = [revised_plan, revised_plan]
        
        self.mock_tools.execute_tool = AsyncMock(return_value=[])
        self.mock_report.generate_report.return_value = "Report"
        
        # Run
        asyncio.run(self.agent.process_query("Dynamic test"))
        
        # Verify tool execution count
        # Should be 2: 1 initial + 1 added dynamically
        self.assertEqual(self.mock_tools.execute_tool.call_count, 2)
        self.mock_tools.execute_tool.assert_any_call("search_arxiv", query="new query")
        print("\n✓ Agent dynamically added and executed a new step based on reflection!")

if __name__ == '__main__':
    unittest.main()
