"""LLMFlow-Search — LangGraph web research agent for macOS with Ollama.

The agent is split across submodules (``nodes``, ``llm``, ``sources``, ``memory``,
...). ``cli`` is the console-script entry point; the import of the app module (which
pulls in langgraph, mcp, and the footnote_mcp helper) is deferred into the call so
``import llmflow_search`` stays cheap.
"""

import os
import subprocess
import sys
import warnings
from pathlib import Path

# langgraph/pydantic emit this on import of the graph machinery; it is not actionable here.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version.*",
    category=Warning,
)


def _ensure_default_browser() -> None:
    """Install Chromium when the packaged default MCP server needs it."""
    server_cmd = os.getenv("LLMFLOW_SEARCH_MCP_CMD", "footnote-mcp").split()
    if not server_cmd or Path(server_cmd[0]).name != "footnote-mcp":
        return

    result = subprocess.run(
        [sys.executable, "-m", "playwright", "install", "chromium"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        print(
            "[!] Chromium installation failed; browser-backed MCP tools may be unavailable.",
            file=sys.stderr,
        )


def cli():
    import asyncio

    from .app import main

    _ensure_default_browser()
    asyncio.run(main())
