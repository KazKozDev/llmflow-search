import aiohttp
from typing import Any, Dict, List
from .base import BaseTool
from .tool_utils import retry_with_backoff, format_error, measure_time, DEFAULT_TIMEOUT

class GutenbergTool(BaseTool):
    """Tool for searching Project Gutenberg books with timeout and retry."""
    
    def __init__(self):
        super().__init__(
            name="search_gutenberg",
            description="Search for free books on Project Gutenberg."
        )
        self.api_url = "https://gutendex.com/books"
    
    @measure_time
    async def execute(self, query: str, limit: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously with retry logic.
        
        Args:
            query: Search query
            limit: Number of results (default: 5)
            
        Returns:
            List of book dictionaries or error dict
        """
        params = {'search': query}
        
        try:
            async def _fetch():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.api_url,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Gutendex returned {response.status}")
                        return await response.json()
            
            # Retry with backoff
            data = await retry_with_backoff(_fetch)
            results = data.get('results', [])
            
            if not results:
                return [{"error": "No results found", "query": query}]
            
            normalized_results = []
            for item in results[:limit]:
                authors = ", ".join([a.get('name', '') for a in item.get('authors', [])[:2]])
                if len(item.get('authors', [])) > 2:
                    authors += " et al."
                
                languages = ", ".join(item.get('languages', []))
                downloads = item.get('download_count', 0)
                
                normalized_results.append({
                    "title": item.get('title', 'Unknown'),
                    "url": f"https://www.gutenberg.org/ebooks/{item.get('id')}",
                    "snippet": f"Authors: {authors or 'Unknown'}. Languages: {languages or 'N/A'}. Downloads: {downloads}",
                    "subjects": item.get('subjects', [])[:3],  # Limit subjects
                    "source": "gutenberg"
                })
            
            return normalized_results
            
        except Exception as e:
            return format_error(e, "Gutenberg", query)
