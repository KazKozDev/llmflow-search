"""
Tool Usage Tests - Verify agent can use all available tools.
"""
import pytest
import asyncio
from core.agent_factory import AgentFactory


# Test queries designed to trigger specific tools
TOOL_TEST_QUERIES = {
    "search_duckduckgo": "What is Python programming language",
    "search_wikipedia": "Biography of Albert Einstein",
    "search_arxiv": "Recent machine learning papers on transformers",
    "search_pubmed": "Medical research on diabetes treatment",
    "search_gutenberg": "Books by Jane Austen",
    "search_openstreetmap": "Where is Eiffel Tower located",
    "search_youtube": "Python tutorial for beginners video",
    "search_wayback": "Historical versions of google.com website",
    "search_searxng": "Complex research query about quantum computing applications"
}


@pytest.mark.asyncio
async def test_planning_module_understands_queries():
    """Test that Planning Module (via LLM) understands query intent and selects tools."""
    from core.planning_module import PlanningModule
    from core.llm_gateway import LLMGateway
    from core.llm_service import LLMService
    from core.config import load_config
    
    config = load_config()
    llm_service = LLMService(config)
    llm_gateway = LLMGateway(llm_service)
    planning = PlanningModule(llm_gateway)
    
    test_cases = [
        ("Find ArXiv papers on neural networks", "search_arxiv"),
        ("Medical research on diabetes", "search_pubmed"),
        ("Books by Shakespeare", "search_gutenberg"),
        ("Where is Paris", "search_openstreetmap"),
        ("Python tutorial video", "search_youtube"),
    ]
    
    results = []
    for query, expected_tool in test_cases:
        plan = await planning.create_plan(query)
        tools_used = [step["type"] for step in plan["steps"]]
        
        has_expected = expected_tool in tools_used
        results.append({
            "query": query,
            "expected": expected_tool,
            "tools_planned": tools_used,
            "passed": has_expected
        })
        
        if has_expected:
            print(f"✓ {query} → {expected_tool}")
        else:
            print(f"✗ {query} → Expected {expected_tool}, got {tools_used}")
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\nPlanning Module: {passed}/{total} tests passed")
    
    return results


@pytest.mark.asyncio
async def test_tools_execution():
    """Test that each tool can execute successfully."""
    from core.tools_module import ToolsModule
    from core.memory_module import MemoryModule
    from core.llm_gateway import LLMGateway
    from core.llm_service import LLMService
    from core.config import load_config
    
    config = load_config()
    memory = MemoryModule()
    llm_service = LLMService(config)
    llm_gateway = LLMGateway(llm_service)
    
    tools = ToolsModule(
        memory=memory,
        llm_service=llm_gateway,
        config=config.model_dump(),
        max_results=5
    )
    
    results = []
    for tool_name, query in TOOL_TEST_QUERIES.items():
        try:
            result = await tools.execute_tool(tool_name, query=query)
            
            # Check if result is valid
            is_valid = (
                result and
                not str(result).startswith("Error") and
                len(str(result)) > 10
            )
            
            results.append({
                "tool": tool_name,
                "query": query,
                "passed": is_valid,
                "result_length": len(str(result)) if result else 0
            })
            
            if is_valid:
                print(f"✓ {tool_name}: {len(str(result))} chars")
            else:
                print(f"✗ {tool_name}: Failed or empty result")
                
        except Exception as e:
            results.append({
                "tool": tool_name,
                "query": query,
                "passed": False,
                "error": str(e)
            })
            print(f"✗ {tool_name}: Exception - {e}")
    
    # Summary
    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\nTool Execution: {passed}/{total} tools working")
    
    return results


@pytest.mark.asyncio
async def test_agent_integration():
    """Test that agent actually uses tools during search."""
    agent = await AgentFactory.create_agent(max_iterations=5)
    
    # Test a query that should trigger ArXiv
    query = "Find 3 recent papers on large language models"
    report = await agent.process_query(query)
    
    # Check memory for tool usage
    memory_data = agent.memory.get_long_term()
    tools_used = []
    
    for entry in memory_data:
        if isinstance(entry, dict) and "tools_used" in entry:
            tools_used.extend(entry["tools_used"])
    
    print(f"\nAgent used tools: {tools_used}")
    print(f"Report length: {len(report)} chars")
    
    # Verify ArXiv was used or at least some tool was used
    assert len(tools_used) > 0, "Agent didn't use any tools!"
    
    return {
        "tools_used": tools_used,
        "report_length": len(report)
    }


async def run_all_tests():
    """Run all tool tests."""
    print("=" * 60)
    print("TOOL USAGE TEST SUITE")
    print("=" * 60)
    
    print("\n1. Testing Query Parser...")
    await test_query_parser()
    
    print("\n2. Testing Planning Module Tool Selection...")
    planning_results = await test_planning_module_tool_selection()
    
    print("\n3. Testing Individual Tool Execution...")
    execution_results = await test_tools_execution()
    
    print("\n4. Testing Agent Integration...")
    integration_result = await test_agent_integration()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    # Planning summary
    planning_passed = sum(1 for r in planning_results if r["passed"])
    print(f"Planning: {planning_passed}/{len(planning_results)} queries planned correctly")
    
    # Execution summary
    exec_passed = sum(1 for r in execution_results if r["passed"])
    print(f"Execution: {exec_passed}/{len(execution_results)} tools executed successfully")
    
    # Integration
    print(f"Integration: Agent used {len(integration_result['tools_used'])} tool(s)")
    
    return {
        "planning": planning_results,
        "execution": execution_results,
        "integration": integration_result
    }


if __name__ == "__main__":
    # Run tests
    results = asyncio.run(run_all_tests())
    
    # Save detailed results
    import json
    with open("test_results_tools.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\nDetailed results saved to test_results_tools.json")
