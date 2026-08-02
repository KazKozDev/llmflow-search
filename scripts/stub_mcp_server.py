#!/usr/bin/env python3
"""A tiny stub MCP server for exercising LLMFlow-Search's generic profile by hand.

It exposes one trivial tool (`get_fact`) and nothing footnote-specific, so connecting
LLMFlow-Search to it forces the generic profile. Drive the agent against it with:

    LLMFLOW_SEARCH_MCP_CMD="python scripts/stub_mcp_server.py" PYTHONPATH=src python -m llmflow_search

then ask e.g. "what is the capital of France?".
"""
import asyncio

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import ServerCapabilities, TextContent, Tool

server = Server("stub")

_FACTS = {
    "capital of france": "The capital of France is Paris.",
    "speed of light": "The speed of light in vacuum is 299,792,458 metres per second.",
    "tallest mountain": "Mount Everest is the tallest mountain above sea level, at 8,849 metres.",
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_fact",
            description="Look up a short, sourced factual statement about a topic.",
            inputSchema={
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "Topic to look up"}},
                "required": ["topic"],
            },
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_fact":
        topic = (arguments.get("topic") or "").strip().lower()
        fact = next((v for k, v in _FACTS.items() if k in topic or topic in k), None)
        return [TextContent(type="text", text=fact or f"No fact on file for '{topic}'.")]
    return [TextContent(type="text", text=f"unknown tool: {name}")]


async def _main():
    init = InitializationOptions(
        server_name="stub",
        server_version="0.1.0",
        capabilities=ServerCapabilities(tools={}),
    )
    async with stdio_server() as (read, write):
        await server.run(read, write, init)


if __name__ == "__main__":
    asyncio.run(_main())
