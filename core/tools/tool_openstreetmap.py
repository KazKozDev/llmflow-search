import aiohttp
from typing import Any, Dict, List
from .base import BaseTool
from .tool_utils import retry_with_backoff, format_error, measure_time, DEFAULT_TIMEOUT

class OpenStreetMapTool(BaseTool):
    """Tool for geocoding via OpenStreetMap with timeout and retry."""
    
    def __init__(self):
        super().__init__(
            name="search_openstreetmap",
            description="Search for locations and coordinates via OpenStreetMap."
        )
        self.api_url = "https://nominatim.openstreetmap.org/search"
        self.headers = {
            'User-Agent': 'LLMFlow-Search/1.0 (https://github.com/llmflow; research@example.com)'
        }
    
    @measure_time
    async def execute(self, query: str, limit: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously with retry logic.
        
        Args:
            query: Location search query
            limit: Number of results (default: 5)
            
        Returns:
            List of location dictionaries or error dict
        """
        params = {
            'q': query,
            'format': 'json',
            'limit': limit
        }
        
        try:
            async def _fetch():
                async with aiohttp.ClientSession(headers=self.headers) as session:
                    async with session.get(
                        self.api_url,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Nominatim returned {response.status}")
                        return await response.json()
            
            # Retry with backoff
            data = await retry_with_backoff(_fetch)
            
            if not data:
                return [{"error": "No results found", "query": query}]
            
            results = []
            for item in data:
                results.append({
                    "title": item.get('display_name', 'Unknown'),
                    "url": f"https://www.openstreetmap.org/{item.get('osm_type')}/{item.get('osm_id')}",
                    "snippet": f"Type: {item.get('type', 'N/A')}, Lat: {item.get('lat', 'N/A')}, Lon: {item.get('lon', 'N/A')}",
                    "lat": item.get('lat'),
                    "lon": item.get('lon'),
                    "type": item.get('type'),
                    "source": "openstreetmap"
                })
            
            return results
            
        except Exception as e:
            return format_error(e, "OpenStreetMap", query)
