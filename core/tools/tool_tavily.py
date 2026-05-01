import os
from typing import Any, Dict, List
from .base import BaseTool


class TavilyTool(BaseTool):
    """Tool for searching the web using Tavily API."""

    def __init__(self):
        super().__init__(
            name="search_tavily",
            description="Search the web using Tavily API."
        )
        from tavily import AsyncTavilyClient
        self.client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    async def execute(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously.

        Args:
            query: The search query

        Returns:
            List of search results
        """
        response = await self.client.search(
            query=query,
            max_results=kwargs.get("max_results", 5),
            search_depth="basic",
        )

        normalized_results = []
        for item in response.get("results", []):
            normalized_results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
                "source": "tavily",
            })

        return normalized_results
