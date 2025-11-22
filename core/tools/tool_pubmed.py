import aiohttp
from typing import Any, Dict, List
from .base import BaseTool
from .tool_utils import retry_with_backoff, format_error, measure_time, DEFAULT_TIMEOUT

class PubMedTool(BaseTool):
    """Tool for searching biomedical papers on PubMed with timeout and retry."""
    
    def __init__(self):
        super().__init__(
            name="search_pubmed",
            description="Search for biomedical literature on PubMed."
        )
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    @measure_time
    async def execute(self, query: str, max_results: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """
        Execute the search asynchronously with retry logic.
        
        Args:
            query: Search query
            max_results: Maximum results (default: 5)
            
        Returns:
            List of paper dictionaries or error dict
        """
        try:
            # Step 1: Search for IDs with retry
            async def _search_ids():
                search_url = f"{self.base_url}/esearch.fcgi"
                search_params = {
                    'db': 'pubmed',
                    'term': query,
                    'retmode': 'json',
                    'retmax': max_results
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        search_url,
                        params=search_params,
                        timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"PubMed Search returned {response.status}")
                        return await response.json()
            
            data = await retry_with_backoff(_search_ids)
            id_list = data.get('esearchresult', {}).get('idlist', [])
            
            if not id_list:
                return [{"error": "No results found", "query": query}]
            
            # Step 2: Fetch details for IDs with retry
            async def _fetch_summaries():
                fetch_url = f"{self.base_url}/esummary.fcgi"
                fetch_params = {
                    'db': 'pubmed',
                    'id': ",".join(id_list),
                    'retmode': 'json'
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        fetch_url,
                        params=fetch_params,
                        timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
                    ) as response:
                        if response.status != 200:
                            raise Exception(f"PubMed Summary returned {response.status}")
                        return await response.json()
            
            data = await retry_with_backoff(_fetch_summaries)
            result_dict = data.get('result', {})
            
            results = []
            for uid in id_list:
                item = result_dict.get(uid)
                if item:
                    authors = ", ".join([a.get('name', '') for a in item.get('authors', [])[:3]])
                    if len(item.get('authors', [])) > 3:
                        authors += " et al."
                    
                    results.append({
                        "title": item.get('title', 'Unknown'),
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/",
                        "snippet": f"Authors: {authors}. Source: {item.get('source', 'N/A')} ({item.get('pubdate', 'N/A')})",
                        "uid": uid,
                        "source": "pubmed"
                    })
            
            return results if results else [{"error": "No valid results", "query": query}]
            
        except Exception as e:
            return format_error(e, "PubMed", query)
