#!/usr/bin/env python3
"""
LLMFlow Search Agent - Search Tools System
Tools Module - Implements search tools for the agent.
Includes DuckDuckGo search, Wikipedia search, and web page parsing.
"""
import logging
import requests
import time
import json
import re
import sys
import os
from urllib.parse import urlencode, quote_plus
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

# Add current directory to path for importing tools
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Import DuckDuckGoSearcher from tools directory
from tools.tool_search_duckduckgo import DuckDuckGoSearcher, default_searcher

class ToolsModule:
    def __init__(self, memory, llm_service, max_results=5, 
                 safe_search=True, parse_top_results=3):
        """
        Initialize the tools module.
        
        Args:
            memory: Memory module for storing results
            llm_service: LLM service for analysis
            max_results: Maximum number of search results to return
            safe_search: Whether to enable safe search
            parse_top_results: Number of top results to consider for parsing
        """
        self.memory = memory
        self.llm_service = llm_service
        self.max_results = max_results
        self.safe_search = safe_search
        self.parse_top_results = parse_top_results
        
        self.logger = logging.getLogger(__name__)
        
        # User agent for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://duckduckgo.com/'
        }
    
    @retry(
        retry=retry_if_exception_type((
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def search_duckduckgo(self, query):
        """
        Search DuckDuckGo for the given query using the enhanced DuckDuckGoSearcher.
        """
        self.logger.info(f"Searching DuckDuckGo for: {query}")
        
        try:
            # Check if query contains Cyrillic characters (Russian)
            is_cyrillic = bool(re.search('[а-яА-Я]', query))
            
            if default_searcher is not None:
                searcher = default_searcher
            else:
                searcher = DuckDuckGoSearcher(use_cache=True, verbose=False)
            
            raw_results = searcher.search(query)
            
            results = []
            for item in raw_results:
                results.append({
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "snippet": item.get("content", ""),  # Add snippet field for compatibility
                    "url": item.get("link", "") or item.get("url", "")
                })
            
            self.logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            self.logger.error(f"Error searching DuckDuckGo: {e}")
            return []
    
    # The _fallback_search method has been removed as the enhanced search module is now used
    
    @retry(
        retry=retry_if_exception_type((
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def search_wikipedia(self, query):
        """
        Search Wikipedia.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with Wikipedia page information
        """
        self.logger.info(f"Searching Wikipedia for: {query}")
        
        # URL-encode the query
        encoded_query = quote_plus(query)
        
        # Try direct page access first
        wiki_url = f"https://en.wikipedia.org/wiki/{encoded_query}"
        
        try:
            # Make request
            response = requests.get(
                wiki_url, 
                headers=self.headers,
                timeout=10
            )
            
            # If page doesn't exist, try search
            if response.status_code == 404 or 'Wikipedia does not have an article' in response.text:
                return self._wikipedia_search(query)
            
            # Parse page
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get title
            title = soup.select_one('#firstHeading').get_text(strip=True)
            
            # Get summary (first paragraph)
            content_div = soup.select_one('#mw-content-text')
            paragraphs = content_div.select('p')
            
            summary = ""
            for p in paragraphs:
                if p.get_text(strip=True):
                    summary = p.get_text(strip=True)
                    break
            
            # Get sections
            sections = []
            for heading in soup.select('h2'):
                if not heading.select_one('.mw-headline'):
                    continue
                
                heading_text = heading.select_one('.mw-headline').get_text(strip=True)
                
                # Skip certain sections
                if heading_text in ['References', 'External links', 'See also']:
                    continue
                
                sections.append(heading_text)
            
            self.logger.info(f"Found Wikipedia page: {title}")
            
            return {
                "page_found": True,
                "title": title,
                "url": response.url,
                "summary": summary,
                "sections": sections[:5]  # Limit to top 5 sections
            }
            
        except Exception as e:
            self.logger.error(f"Error accessing Wikipedia page: {str(e)}")
            return self._wikipedia_search(query)
    
    def _wikipedia_search(self, query):
        """
        Perform a Wikipedia search when direct page access fails.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary with search results
        """
        self.logger.info(f"Performing Wikipedia search for: {query}")
        
        search_url = f"https://en.wikipedia.org/w/index.php?search={quote_plus(query)}"
        
        try:
            # Make request
            response = requests.get(
                search_url, 
                headers=self.headers,
                timeout=10
            )
            response.raise_for_status()
            
            # Check if we were redirected to a specific page
            if '/wiki/' in response.url and 'Special:Search' not in response.url:
                # We were redirected to a specific page
                soup = BeautifulSoup(response.text, 'html.parser')
                title = soup.select_one('#firstHeading').get_text(strip=True)
                
                # Get summary
                content_div = soup.select_one('#mw-content-text')
                paragraphs = content_div.select('p')
                
                summary = ""
                for p in paragraphs:
                    if p.get_text(strip=True):
                        summary = p.get_text(strip=True)
                        break
                
                self.logger.info(f"Redirected to Wikipedia page: {title}")
                
                return {
                    "page_found": True,
                    "title": title,
                    "url": response.url,
                    "summary": summary
                }
            
            # Parse search results
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get search results
            results = []
            for result in soup.select('.mw-search-result'):
                title_el = result.select_one('.mw-search-result-heading a')
                snippet_el = result.select_one('.searchresult')
                
                if title_el and snippet_el:
                    title = title_el.get_text(strip=True)
                    snippet = snippet_el.get_text(strip=True)
                    url = f"https://en.wikipedia.org{title_el['href']}"
                    
                    results.append({
                        "title": title,
                        "snippet": snippet,
                        "url": url
                    })
            
            # If we have results, return the top one
            if results:
                top_result = results[0]
                self.logger.info(f"Found Wikipedia search result: {top_result['title']}")
                
                return {
                    "page_found": False,
                    "search_results": results[:3],
                    "suggestion": top_result['title'],
                    "url": top_result['url']
                }
            
            self.logger.info("No Wikipedia results found")
            return {
                "page_found": False,
                "search_results": [],
                "suggestion": None,
                "url": None
            }
            
        except Exception as e:
            self.logger.error(f"Error searching Wikipedia: {str(e)}")
            return {
                "page_found": False,
                "error": str(e)
            }
    
    @retry(
        retry=retry_if_exception_type((
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError
        )),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def parse_duckduckgo_result(self, url):
        """
        Parse the content of a DuckDuckGo search result URL.
        
        Args:
            url: The URL to parse
            
        Returns:
            Parsed content as markdown
        """
        self.logger.info(f"Parsing DuckDuckGo result: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            # Try to extract main content
            article = soup.find('article')
            if article:
                content_html = str(article)
            else:
                # Fallback: get the largest <div> by text length
                divs = soup.find_all('div')
                content_html = max(divs, key=lambda d: len(d.get_text(strip=True)), default="")
                content_html = str(content_html)
            # Convert HTML to markdown
            markdown = md(content_html)
            # Truncate if too long
            if len(markdown) > 8000:
                markdown = markdown[:8000] + "\n..."
            return markdown
        except Exception as e:
            self.logger.error(f"Error parsing URL {url}: {str(e)}")
            return f"[Error parsing {url}: {str(e)}]"
