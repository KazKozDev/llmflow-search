import aiohttp
from typing import Any, Dict, List
from .base import BaseTool
from .tool_utils import retry_with_backoff, format_error, fetch_with_timeout, measure_time, DEFAULT_TIMEOUT

class ArXivTool(BaseTool):
    """Tool for searching scientific papers on ArXiv using async API."""
    
    def __init__(self):
        super().__init__(
            name="search_arxiv",
            description="Search for scientific papers on ArXiv."
        )
        self.api_url = "http://export.arxiv.org/api/query"
    
    @measure_time
    async def execute(self, query: str, max_results: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously using ArXiv API.
        
        Args:
            query: Search query
            max_results: Maximum number of results (default: 5)
            
        Returns:
            List of paper dictionaries or error dict
        """
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            async def _fetch():
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        self.api_url,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"ArXiv API returned {response.status}")
                        return await response.text()
            
            # Retry with backoff
            xml_data = await retry_with_backoff(_fetch)
            
            # Parse XML response
            import xml.etree.ElementTree as ET
            root = ET.fromstring(xml_data)
            
            # ArXiv uses Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            results = []
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                published = entry.find('atom:published', ns)
                link = entry.find('atom:id', ns)
                
                # Get authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None and name.text:
                        authors.append(name.text)
                
                # Get category (primary subject)
                category = entry.find('arxiv:primary_category', ns)
                subject = category.get('term') if category is not None else 'N/A'
                
                results.append({
                    "title": title.text.strip() if title is not None else "Unknown",
                    "url": link.text.strip() if link is not None else "",
                    "snippet": (summary.text.strip()[:400] + "...") if summary is not None and summary.text else "No abstract available",
                    "published": published.text[:10] if published is not None else "Unknown",
                    "authors": authors[:3],  # Limit to first 3 authors
                    "subject": subject,
                    "source": "arxiv"
                })
            
            return results if results else [{"error": "No results found", "query": query}]
            
        except Exception as e:
            return format_error(e, "ArXiv", query)
