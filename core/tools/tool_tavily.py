import os
import logging
from typing import Any, Dict, List
from .base import BaseTool
from .tool_utils import format_error, measure_time

logger = logging.getLogger(__name__)


class TavilyTool(BaseTool):
    """Tool for searching via the Tavily API."""

    def __init__(self):
        super().__init__(
            name="search_tavily",
            description="Search the web using Tavily API."
        )
        self.api_key = os.environ.get("TAVILY_API_KEY", "")

    @measure_time
    async def execute(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute a Tavily search.

        Args:
            query: Search query
            limit: Number of results (default: 10)

        Returns:
            List of result dictionaries or error dict
        """
        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=self.api_key)
            response = await client.search(
                query=query,
                max_results=limit,
                search_depth="basic",
            )

            results = response.get("results", [])
            if not results:
                return [{"error": "No results found", "query": query}]

            normalized_results = []
            for item in results[:limit]:
                normalized_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": "tavily"
                })

            return normalized_results

        except Exception as e:
            return format_error(e, "Tavily", query)
