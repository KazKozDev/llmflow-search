import asyncio
from typing import Any, Dict
from .base import BaseTool
from .impl_wikipedia import WikipediaTool as OriginalWikipediaTool

class WikipediaTool(BaseTool):
    """Tool for searching Wikipedia asynchronously."""
    
    def __init__(self):
        super().__init__(
            name="search_wikipedia",
            description="Search Wikipedia for a query."
        )
        self.wiki = OriginalWikipediaTool()
    
    async def execute(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the search asynchronously.
        
        Args:
            query: The search query
            
        Returns:
            Wikipedia search result
        """
        # The original tool has an async search_wikipedia method
        # But it returns a list of results. We need to match the expected format of the AgentCore
        # which expects a single result object with 'page_found', etc.
        # So we'll reimplement the logic here using the async methods of the original tool
        
        try:
            # Try to get summary directly (simulating "I'm feeling lucky")
            try:
                summary = await self.wiki.get_article_summary(query)
                return {
                    "page_found": True,
                    "title": summary.get('title'),
                    "url": summary.get('url'),
                    "summary": summary.get('extract'),
                    "sections": [] # We don't get sections in summary
                }
            except Exception:
                # If direct lookup fails, search
                results = await self.wiki.search_wikipedia(query, limit=3)
                if results:
                    top_result = results[0]
                    return {
                        "page_found": False,
                        "search_results": results,
                        "suggestion": top_result['title'],
                        "url": f"https://en.wikipedia.org/?curid={top_result['pageid']}"
                    }
                else:
                    return {
                        "page_found": False,
                        "search_results": [],
                        "suggestion": None,
                        "url": None
                    }
                    
        except Exception as e:
            return {
                "page_found": False,
                "error": str(e)
            }
