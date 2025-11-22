#!/usr/bin/env python3
"""
LLMFlow Search Agent - Search Tools System
Tools Module - Implements search tools for the agent.
Includes DuckDuckGo search, Wikipedia search, and web page parsing.
"""
import logging
import aiohttp
import asyncio
import os
import sys
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Add current directory to path for importing tools
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from core.tools.base import BaseTool
from core.tools.tool_duckduckgo import DuckDuckGoTool
from core.tools.tool_wikipedia import WikipediaTool
from core.tools.tool_searxng import SearXNGTool
from core.tools.tool_arxiv import ArXivTool
from core.tools.tool_wayback import WaybackTool
from core.tools.tool_openstreetmap import OpenStreetMapTool
from core.tools.tool_youtube import YouTubeTool
from core.tools.tool_gutenberg import GutenbergTool
from core.tools.tool_pubmed import PubMedTool

class ToolsModule:
    def __init__(self, memory, llm_service, config: dict, max_results=5, 
                 safe_search=True, parse_top_results=3):
        """
        Initialize the tools module.
        
        Args:
            memory: Memory module for storing results
            llm_service: LLM service for analysis
            config: Full configuration dictionary
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
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Initialize cache
        from core.caching.factory import CacheFactory
        cache_config = config.get('cache', {'provider': 'sqlite', 'sqlite_path': './data/cache.db'})
        self.cache = CacheFactory.create(cache_config)
        self.logger.info("Cache initialized")
        
        # Initialize rate limiter
        from core.rate_limiter import RateLimiter
        rate_limit_config = config.get('rate_limits', {'default': {'requests_per_minute': 30}})
        self.rate_limiter = RateLimiter(rate_limit_config)
        self.logger.info("Rate limiter initialized")
        
        # Register tools
        self.tools = {}
        self.register_tool(DuckDuckGoTool())
        self.register_tool(WikipediaTool())
        self.register_tool(SearXNGTool())
        self.register_tool(ArXivTool())
        self.register_tool(WaybackTool())
        self.register_tool(OpenStreetMapTool())
        self.register_tool(YouTubeTool())
        self.register_tool(GutenbergTool())
        self.register_tool(PubMedTool())
        
    def register_tool(self, tool: BaseTool):
        """Register a tool."""
        self.tools[tool.name] = tool
        self.logger.info(f"Registered tool: {tool.name}")
        
    def get_tool(self, name: str) -> BaseTool:
        """Get a tool by name."""
        return self.tools.get(name)
        
    async def execute_tool(self, name: str, **kwargs):
        """Execute a tool by name with caching and rate limiting."""
        # Check cache first
        cache_key = f"{name}:{str(kwargs)}"
        cached_result = await self.cache.get(cache_key)
        if cached_result is not None:
            self.logger.debug(f"Cache hit for {name}")
            return cached_result
        
        # Apply rate limiting
        await self.rate_limiter.acquire(name, wait=True)
        
        # Execute tool
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        
        result = await tool.execute(**kwargs)
        
        # Cache result
        await self.cache.set(cache_key, result)
        
        return result

    async def search_duckduckgo(self, query):
        """
        Search DuckDuckGo for the given query.
        Wrapper for backward compatibility.
        """
        return await self.execute_tool("search_duckduckgo", query=query)
    
    async def search_wikipedia(self, query):
        """
        Search Wikipedia.
        Wrapper for backward compatibility.
        """
        return await self.execute_tool("search_wikipedia", query=query)
            
    async def parse_duckduckgo_result(self, result: dict) -> str:
        """Parse content from a DuckDuckGo search result with caching."""
        from core.parsing_cache import get_parsing_cache
        
        url = result.get("url", "")
        if not url:
            return ""
        
        # Check parsing cache first
        parsing_cache = get_parsing_cache()
        cached_content = parsing_cache.get(url)
        if cached_content:
            self.logger.info(f"Using cached parsing for: {url}")
            return cached_content
        
        # Not in cache - parse it
        self.logger.info(f"Parsing (not cached): {url}")
        
        parsed_content = ""
        
        # Check if it's Wikipedia
        if "wikipedia.org" in url:
            # Use Wikipedia API for better results
            from core.tools.impl_wikipedia import WikipediaToolForLLM
            wiki_tool = WikipediaToolForLLM()
            
            try:
                # Extract article title from URL
                import re
                match = re.search(r'/wiki/([^#?]+)', url)
                if match:
                    title = match.group(1).replace('_', ' ')
                    content = await wiki_tool.async_get_page_content(title)
                    if content:
                        parsed_content = content[:5000]  # Limit content
            except Exception as e:
                self.logger.error(f"Wikipedia parsing error: {e}")
        
        # For non-Wikipedia or if Wikipedia failed, use general parser
        if not parsed_content:
            from core.tools.async_link_parser import extract_content_from_url_async
            try:
                parsed_content = await extract_content_from_url_async(url)
            except Exception as e:
                self.logger.error(f"Error parsing {url}: {e}")
        
        # Cache the result (even if empty)
        if parsed_content:
            parsing_cache.set(url, parsed_content)
        
        return parsed_content
    
    async def _parse_wikipedia_url(self, url):
        """
        Parse Wikipedia URL using the Wikipedia API.
        
        Args:
            url: Wikipedia article URL
            
        Returns:
            Article text content
        """
        try:
            # Extract article title from URL
            # Format: https://en.wikipedia.org/wiki/Article_Title
            import re
            from urllib.parse import unquote
            
            match = re.search(r'wikipedia\.org/wiki/(.+)', url)
            if not match:
                self.logger.warning(f"Could not extract title from Wikipedia URL: {url}")
                return ""
            
            title = unquote(match.group(1))
            
            # Detect language from URL
            lang_match = re.search(r'https?://([a-z]{2})\.wikipedia\.org', url)
            language = lang_match.group(1) if lang_match else 'en'
            
            self.logger.info(f"Fetching Wikipedia article: {title} (lang: {language})")
            
            # Use WikipediaTool to get content via API
            wiki_tool = self.get_tool('search_wikipedia')
            if wiki_tool:
                result = await wiki_tool.execute(query=title, language=language)
                
                # Extract content from result
                if result and result.get('page_found'):
                    content = result.get('content', '')
                    return content
            
            self.logger.warning(f"Could not fetch Wikipedia content for: {title}")
            return ""
            
        except Exception as e:
            self.logger.error(f"Error parsing Wikipedia URL {url}: {e}")
            return ""

    def _parse_html(self, html_content):
        """Helper to parse HTML to Markdown (CPU bound)."""
        soup = BeautifulSoup(html_content, 'html.parser')
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
