import os
import logging
import aiohttp
from typing import Any, Dict, List
from .base import BaseTool
from .tool_utils import retry_with_backoff, format_error, measure_time, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


class SearXNGTool(BaseTool):
    """Tool for searching via SearXNG with timeout and retry."""

    def __init__(self, instance_url="https://searx.be", searxng_tavily_fallback=None):
        super().__init__(
            name="search_searxng",
            description="Search multiple engines via SearXNG."
        )
        self.instance_url = instance_url
        if searxng_tavily_fallback is not None:
            self._tavily_fallback = searxng_tavily_fallback
        else:
            self._tavily_fallback = self._load_fallback_flag()

    @staticmethod
    def _load_fallback_flag() -> bool:
        """Load searxng_tavily_fallback flag from config."""
        try:
            from ..config import load_config
            config = load_config()
            return config.search.searxng_tavily_fallback
        except Exception:
            return False

    @measure_time
    async def execute(self, query: str, limit: int = 10, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously with retry logic.

        Args:
            query: Search query
            limit: Number of results (default: 10)

        Returns:
            List of result dictionaries or error dict
        """
        params = {
            'q': query,
            'format': 'json',
            'language': 'auto'
        }

        try:
            async def _fetch():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.instance_url}/search",
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"SearXNG returned status {response.status}")
                        return await response.json()

            # Retry with backoff
            data = await retry_with_backoff(_fetch)
            results = data.get('results', [])

            if not results:
                return [{"error": "No results found", "query": query}]

            normalized_results = []
            for item in results[:limit]:
                normalized_results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": "searxng"
                })

            return normalized_results

        except Exception as e:
            # Attempt Tavily fallback if configured and API key is available
            if os.environ.get("TAVILY_API_KEY") and self._tavily_fallback:
                logger.warning(
                    f"SearXNG failed for query '{query}', falling back to Tavily: {e}"
                )
                try:
                    from .tool_tavily import TavilyTool
                    tavily_tool = TavilyTool()
                    return await tavily_tool.execute(query=query, limit=limit, **kwargs)
                except Exception as tavily_err:
                    logger.error(f"Tavily fallback also failed: {tavily_err}")
                    return format_error(
                        Exception(f"SearXNG failed: {e}; Tavily fallback also failed: {tavily_err}"),
                        "SearXNG+Tavily",
                        query
                    )
            return format_error(e, "SearXNG", query)
