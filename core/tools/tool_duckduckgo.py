import asyncio
from typing import Any, Dict, List
from .base import BaseTool
from .impl_duckduckgo import DuckDuckGoSearcher

class DuckDuckGoTool(BaseTool):
    """Tool for searching DuckDuckGo asynchronously."""
    
    def __init__(self):
        super().__init__(
            name="search_duckduckgo",
            description="Search DuckDuckGo for a query."
        )
        self.searcher = DuckDuckGoSearcher(use_cache=True, verbose=False)
    
    async def execute(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously.
        
        Args:
            query: The search query
            
        Returns:
            List of search results
        """
        loop = asyncio.get_event_loop()
        # Run the blocking search in an executor
        results = await loop.run_in_executor(None, self.searcher.search, query)
        
        # Normalize results
        normalized_results = []
        for item in results:
            normalized_results.append({
                "title": item.get("title", ""),
                "content": item.get("content", ""),
                "snippet": item.get("content", ""),
                "url": item.get("link", "") or item.get("url", "")
            })
            
        return normalized_results
