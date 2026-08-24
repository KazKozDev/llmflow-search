"""LLMFlow-Search — LangGraph web research agent for local Ollama.

The agent is split across submodules (``nodes``, ``llm``, ``sources``, ``memory``,
...). ``cli`` is the console-script entry point; the import of the app module (which
pulls in langgraph, mcp, and the footnote_mcp helper) is deferred into the call so
``import llmflow_search`` stays cheap.
"""

import warnings

# langgraph/pydantic emit this on import of the graph machinery; it is not actionable here.
warnings.filterwarnings(
    "ignore",
    message=r"The default value of `allowed_objects` will change in a future version.*",
    category=Warning,
)


def cli():
    import asyncio

    from .app import main

    asyncio.run(main())
