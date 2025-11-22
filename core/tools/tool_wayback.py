import aiohttp
from typing import Any, Dict, List
from .base import BaseTool
from .tool_utils import retry_with_backoff, format_error, measure_time, DEFAULT_TIMEOUT

class WaybackTool(BaseTool):
    """
    Tool for checking Internet Archive (Wayback Machine).
    Note: This tool requires a URL as input, not a text query.
    """
    
    def __init__(self):
        super().__init__(
            name="search_wayback",
            description="Check if URLs are archived in the Wayback Machine. Requires URL as input."
        )
        self.api_url = "http://archive.org/wayback/available"
    
    @measure_time
    async def execute(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously with retry logic.
        
        Args:
            query: URL to check (must start with http/https)
            
        Returns:
            List with archive info or error dict
        """
        # Extract URL from query if it's mixed text
        url = self._extract_url(query)
        
        if not url:
            return [{
                "error": "No valid URL found in query",
                "hint": "Wayback Machine requires a URL like https://example.com",
                "query": query
            }]
        
        try:
            async def _fetch():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.api_url,
                        params={'url': url},
                        timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"Wayback API returned {response.status}")
                        return await response.json()
            
            # Retry with backoff
            data = await retry_with_backoff(_fetch)
            snapshots = data.get('archived_snapshots', {})
            closest = snapshots.get('closest', {})
            
            if closest and closest.get('available'):
                timestamp = closest.get('timestamp', '')
                # Format timestamp: YYYYMMDDHHMMSS -> YYYY-MM-DD
                formatted_date = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}" if len(timestamp) >= 8 else timestamp
                
                return [{
                    "title": f"Archived: {url}",
                    "url": closest.get('url', ''),
                    "snippet": f"Snapshot from {formatted_date}. Status: {closest.get('status', 'N/A')}",
                    "available": True,
                    "timestamp": timestamp,
                    "original_url": url,
                    "source": "wayback"
                }]
            else:
                return [{
                    "available": False,
                    "snippet": f"No archived snapshots found for {url}",
                    "original_url": url,
                    "source": "wayback"
                }]
                
        except Exception as e:
            return format_error(e, "Wayback Machine", query)
    
    def _extract_url(self, text: str) -> str:
        """
        Extract URL from text.
        
        Args:
            text: Text that may contain URL
            
        Returns:
            Extracted URL or empty string
        """
        import re
        
        # If starts with http, use as-is
        if text.strip().startswith('http'):
            return text.strip()
        
        # Try to find URL in text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        match = re.search(url_pattern, text)
        
        return match.group(0) if match else ""
