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
    async def execute(self, query: str, max_results: int = 5, year_filter: Dict[str, int] = None, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously using ArXiv API with pagination support.
        
        Args:
            query: Search query
            max_results: Maximum number of results (default: 5, max: 1000)
            year_filter: Optional year filter {'min': 2020} for filtering by publication year
            
        Returns:
            List of paper dictionaries or error dict
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # ArXiv API limits to 2000 per request, but we'll batch for reliability
        # Also, the API max_results parameter has a hard limit of 2000.
        # We'll cap our internal max_results to 1000 to be safe and efficient.
        max_results = min(max_results, 1000)
        max_per_request = 100 # A reasonable batch size for ArXiv API
        all_results = []
        
        # Calculate number of batches needed
        num_batches = (max_results + max_per_request - 1) // max_per_request
        
        logger.info(f"ArXiv: Fetching {max_results} results for query '{query}' in {num_batches} batch(es)")
        
        try:
            for batch_num in range(num_batches):
                start_index = batch_num * max_per_request
                current_max_results = min(max_per_request, max_results - len(all_results))
                
                if current_max_results <= 0:
                    break # No more results needed
                
                params = {
                    "search_query": query,
                    "start": start_index,
                    "max_results": current_max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending"
                }
                
                logger.debug(f"ArXiv: Requesting batch {batch_num + 1} with params: {params}")
                
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
                
                batch_results = []
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
                    
                    # Parse publication year
                    pub_year = None
                    if published is not None and published.text:
                        try:
                            from datetime import datetime
                            pub_date = datetime.fromisoformat(published.text.replace('Z', '+00:00'))
                            pub_year = pub_date.year
                        except:
                            pass
                    
                    # Apply year filter if specified
                    if year_filter and pub_year:
                        min_year = year_filter.get('min')
                        max_year = year_filter.get('max')
                        if min_year and pub_year < min_year:
                            continue
                        if max_year and pub_year > max_year:
                            continue
                    
                    batch_results.append({
                        "title": title.text.strip() if title is not None else "Unknown",
                        "url": link.text.strip() if link is not None else "",
                        "snippet": (summary.text.strip()[:400] + "...") if summary is not None and summary.text else "No abstract available",
                        "authors": ", ".join(authors[:3]) if authors else "Unknown",
                        "published": published.text[:10] if published is not None and published.text else "Unknown",
                        "year": pub_year,
                        "subject": subject
                    })
                
                all_results.extend(batch_results)
                logger.info(f"ArXiv batch {batch_num + 1}/{num_batches}: collected {len(batch_results)} results")
                
                # Stop if we have enough results
                if len(all_results) >= max_results:
                    break
                
                # Rate limiting: small delay between batches
                if batch_num < num_batches - 1:
                    import asyncio
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"ArXiv API error: {e}")
            # Return what we have, or error
            if all_results:
                return all_results
            return format_error(e, "ArXiv", query)
        
        logger.info(f"ArXiv: Total collected {len(all_results)} results")
        return all_results[:max_results] if all_results else [{"error": "No results found", "query": query}]
