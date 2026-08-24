#!/usr/bin/env python3
"""Live smoke test for the planning agent. Needs ollama + the footnote-mcp server.

Run from the repo root:  PYTHONPATH=src python scripts/live_smoke.py
"""
import asyncio
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
)
from llmflow_search.agent import _default_search_memory, build_graph, load_mcp_tools
from llmflow_search.console import print


async def main():
    tools = await load_mcp_tools()
    print(f"Tools: {len(tools)}")

    graph = build_graph("llama3.2:3b", tools)

    state = {
        "task": "find today's USD exchange rate",
        "requirements_result": {},
        "plan": [],
        "completed_steps": [],
        "scratchpad": "",
        "sources": [],
        "draft_result": {},
        "verification_result": {},
        "final_answer": "",
        "iteration": 0,
        "replan_count": 0,
        "evidence_round": 0,
        "search_memory": _default_search_memory(),
    }

    result = await graph.ainvoke(state, {"recursion_limit": 60})

    print(f"\nSteps: {len(result['completed_steps'])}")
    for c in result["completed_steps"]:
        print(f"  {c['step'][:80]}")

    answer = result.get("final_answer", "")
    if hasattr(answer, "content"):
        answer = answer.content
    print(f"\nAnswer ({len(answer)} chars):\n{answer[:500]}")
    print("\nOK")


asyncio.run(main())
