import aiohttp
from typing import Any, Dict, List
from .base import BaseTool
from .tool_utils import retry_with_backoff, format_error, safe_get, measure_time, DEFAULT_TIMEOUT

class YouTubeTool(BaseTool):
    """Tool for searching YouTube videos with safe error handling."""
    
    def __init__(self):
        super().__init__(
            name="search_youtube",
            description="Search for videos on YouTube."
        )
    
    @measure_time  
    async def execute(self, query: str, limit: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously with safe array access.
        
        Args:
            query: Search query
            limit: Number of results (default: 5)
            
        Returns:
            List of video dictionaries or error dict
        """
        try:
            from youtubesearchpython import VideosSearch
            import asyncio
            
            def _search():
                try:
                    videosSearch = VideosSearch(query, limit=limit)
                    result = videosSearch.result()
                    return result.get('result', [])
                except Exception as e:
                    raise Exception(f"YouTube search failed: {e}")
            
            loop = asyncio.get_event_loop()
            # Run with timeout
            raw_results = await asyncio.wait_for(
                loop.run_in_executor(None, _search),
                timeout=DEFAULT_TIMEOUT
            )
            
            if not raw_results:
                return [{"error": "No results found", "query": query}]
            
            results = []
            for item in raw_results:
                # Safe extraction with fallbacks
                title = safe_get(item, 'title', default='Unknown Title')
                link = safe_get(item, 'link', default='')
                duration = safe_get(item, 'duration', default='N/A')
                
                # Safe viewCount extraction
                view_count_obj = safe_get(item, 'viewCount', default={})
                if isinstance(view_count_obj, dict):
                    views = safe_get(view_count_obj, 'short', default='N/A views')
                else:
                    views = 'N/A views'
                
                # Safe channel extraction
                channel_obj = safe_get(item, 'channel', default={})
                if isinstance(channel_obj, dict):
                    channel = safe_get(channel_obj, 'name', default='Unknown Channel')
                else:
                    channel = 'Unknown Channel'
                
                # Safe description extraction
                desc_snippets = safe_get(item, 'descriptionSnippet', default=[])
                if isinstance(desc_snippets, list) and len(desc_snippets) > 0:
                    description = safe_get(desc_snippets, 0, 'text', default='')
                else:
                    description = ''
                
                snippet = f"Duration: {duration}, Views: {views}"
                if description:
                    snippet += f". {description[:100]}"
                
                results.append({
                    "title": title,
                    "url": link,
                    "snippet": snippet,
                    "channel": channel,
                    "duration": duration,
                    "source": "youtube"
                })
            
            return results
            
        except ImportError:
            return format_error(
                Exception("youtube-search-python library not installed"),
                "YouTube",
                query
            )
        except asyncio.TimeoutError:
            return format_error(
                Exception(f"Search timed out after {DEFAULT_TIMEOUT}s"),
                "YouTube",
                query
            )
        except Exception as e:
            return format_error(e, "YouTube", query)
