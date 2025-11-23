#!/usr/bin/env python3
"""
LLMFlow Search Agent - Wikipedia Search Tool
Implementation of Wikipedia search functionality with content extraction.
Provides direct access to Wikipedia articles and search capabilities.
"""
# tools/wikipedia_tool.py

import requests
import json
import re
import html
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import quote, urlencode
from cachetools import cached, TTLCache
import traceback
import os
import httpx
import asyncio

# Tool Manager import
try:
    from .tool_manager import tool
except ImportError:
    # Fallback if run standalone or tool_manager is not directly importable
    def tool(name=None, description=None, parameters=None):
        def decorator(func):
            # Simple passthrough decorator for fallback
            return func
        return decorator

# Cache for Wikipedia results
# Read TTL from env var, fallback to 21600 (6 hours)
WIKI_CACHE_TTL = int(os.getenv("CACHE_TTL_WIKI", 21600))
wiki_cache = TTLCache(maxsize=100, ttl=WIKI_CACHE_TTL)

class WikipediaTool:
    """
    Tool Name: Wikipedia Information Tool.

    Description: Retrieves information from Wikipedia articles in multiple languages.
    Usage: Can be used to search, get summaries, and extract content from Wikipedia articles.

    System Prompt Addition::

        ```
        You have access to a Wikipedia Tool that can retrieve information from Wikipedia articles.
        When a user asks about facts, definitions, or information that might be found in an encyclopedia,
        use the wikipedia_tool to get this information.

        - To search Wikipedia: Use ``wikipedia_tool.search_wikipedia("quantum physics")``
        - To get a summary: Use ``wikipedia_tool.get_article_summary("Albert Einstein")``
        - To get article content: Use ``wikipedia_tool.get_article_content("Machine Learning")``

        This tool doesn't require any API keys and returns verified information from Wikipedia
        with proper attribution.
        ```
    """
    
    # Tool metadata
    TOOL_NAME = "wikipedia_tool"
    TOOL_DESCRIPTION = "Retrieve information from Wikipedia articles in multiple languages"
    TOOL_PARAMETERS = [
        {"name": "query", "type": "string", "description": "Search term or article title", "required": True},
        {"name": "language", "type": "string", "description": "Wikipedia language code (default: en)", "required": False},
        {"name": "sections", "type": "string", "description": "Specific sections to retrieve (comma separated)", "required": False}
    ]
    TOOL_EXAMPLES = [
        {"query": "What is quantum computing?", "tool_call": "wikipedia_tool.get_article_summary('Quantum computing')"},
        {"query": "Tell me about the history of Rome", "tool_call": "wikipedia_tool.get_article_content('Ancient Rome', sections='History')"},
        {"query": "Who was Marie Curie?", "tool_call": "wikipedia_tool.get_article_summary('Marie Curie')"},
        {"query": "What is the theory of relativity in Russian?", "tool_call": "wikipedia_tool.get_article_summary('Theory of relativity', language='ru')"}
    ]
    
    # Explicit Metadata for LLM
    TOOL_METADATA = {
        "description": "Retrieve information from Wikipedia articles, including summaries, full content, and specific sections.",
        "functions": {
            "search_wikipedia_async": {
                "description": "Searches Wikipedia for articles matching a query and returns a list of titles and snippets.",
                "arguments": ["query", "limit (optional, default=5)"],
                "example": "search_wikipedia_async(query='artificial intelligence', limit=3)"
            },
            "get_article_summary_async": {
                "description": "Fetches a concise summary (usually the introduction) of a specific Wikipedia article.",
                "arguments": ["title"],
                "example": "get_article_summary_async(title='Albert Einstein')"
            },
            "get_article_content_async": {
                "description": "Retrieves the full text content of a Wikipedia article, or specific sections if provided.",
                "arguments": ["title", "section (optional, comma-separated)"],
                "example": "get_article_content_async(title='World War II', section='Causes, Aftermath')"
            }
        }
    }
    
    def __init__(self):
        """Initialize the WikipediaTool with API endpoints."""
        self.api_base_url = "https://{lang}.wikipedia.org/w/api.php"
        
        self.default_language = "en"
        
        self.cache = {}
        self.cache_timestamp = {}
        self.cache_expiry = 3600
        
        
        self.headers = {
            'User-Agent': 'LLMFlow-Search/1.0 (https://github.com/yourusername/llmflow-search; research@example.com) python-httpx/0.25'
        }
        
        self.language = self.default_language
        self.user_agent = self.headers.get('User-Agent')
    
    async def _make_api_request_async(self, lang: str, params: Dict) -> Dict:
        """Helper to make async Wikipedia API requests."""
        url = self.api_base_url.format(lang=lang)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=self.headers, timeout=15)
                response.raise_for_status()
                return response.json()
        except httpx.RequestError as e:
            raise ConnectionError(f"Failed to connect to Wikipedia ({lang}): {e}") from e
        except Exception as e:
            raise Exception(f"Error querying Wikipedia ({lang}): {e}") from e

    async def search_wikipedia(self, query: str, language: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search Wikipedia asynchronously for articles matching a query."""
        print(f"Searching Wikipedia for: {query}")
        if not query or not str(query).strip():
            raise Exception("Error searching Wikipedia: query cannot be empty")
        lang = language if language else self.language
        params = {'action': 'query', 'list': 'search', 'srsearch': query, 'format': 'json', 'srlimit': limit}

        data = await self._make_api_request_async(lang, params)

        try:
            raw_results = data.get('query', {}).get('search', [])
            # Respect limit
            sliced = raw_results[:limit] if limit and isinstance(limit, int) and limit > 0 else raw_results
            results = []
            for item in sliced:
                title = item.get('title', '')
                snippet = self._clean_html(item.get('snippet', ''))
                pageid = item.get('pageid', 0)
                results.append({'title': title, 'snippet': snippet, 'pageid': pageid})
            return results
        except Exception as e:
            raise Exception(f"Error searching Wikipedia: {e}")
    
    async def get_article_summary(self, title: str, language: str = None) -> Dict[str, Any]:
        """
        Get a summary of a Wikipedia article asynchronously.
        
        Args:
            title (str): Article title or search term
            language (str, optional): Wikipedia language code (default: en)
        
        Returns:
            Dict[str, Any]: Article summary with details
            
        Raises:
            Exception: If the API request fails or article not found
        """
        print(f"Getting Wikipedia summary for: {title}")
        
        lang = self._determine_language(language, title)
        
        cache_key = f"summary:{lang}:{title}"
        current_time = datetime.now().timestamp()
        if (cache_key in self.cache and 
            current_time - self.cache_timestamp.get(cache_key, 0) < self.cache_expiry):
            print(f"Using cached summary for {title}")
            return self.cache[cache_key]
        
        # First try the exact title
        try:
            summary_data = await self._get_page_extract_async(title, lang, intro_only=True)
            
            self.cache[cache_key] = summary_data
            self.cache_timestamp[cache_key] = current_time
            
            return summary_data
            
        except Exception as direct_error:
            print(f"Direct title lookup failed: {str(direct_error)}")
            
            # If direct lookup fails, try searching
            try:
                search_results = await self.search_wikipedia(title, lang, limit=1)
                
                # Handle list response format (search_wikipedia returns a list of dicts)
                if isinstance(search_results, list) and len(search_results) > 0:
                    # Use the first search result
                    first_result = search_results[0]
                    actual_title = first_result['title']
                    
                    print(f"Using search result: {actual_title}")
                    
                    # Get the summary using the found title
                    summary_data = await self._get_page_extract_async(actual_title, lang, intro_only=True)
                    
                    # Cache the result
                    self.cache[cache_key] = summary_data
                    self.cache_timestamp[cache_key] = current_time
                    
                    return summary_data
                else:
                    raise ValueError(f"No Wikipedia article found for '{title}'")
                    
            except Exception as search_error:
                error_msg = f"Error getting Wikipedia summary: {str(search_error)}"
                print(error_msg)
                raise Exception(error_msg)
    
    async def get_article_content(self, title: str, language: str = None, sections: str = None) -> Dict[str, Any]:
        """
        Get content from a Wikipedia article asynchronously.
        
        Args:
            title (str): Article title or search term
            language (str, optional): Wikipedia language code (default: en)
            sections (str, optional): Comma-separated list of sections to include
        
        Returns:
            Dict[str, Any]: Article content with details
            
        Raises:
            Exception: If the API request fails or article not found
        """
        print(f"Getting Wikipedia content for: {title}")
        # If pageid provided as int, fetch by pageids
        if isinstance(title, int):
            pageid = title
            lang = language if language else self.language
            params = {'action': 'query', 'prop': 'extracts', 'pageids': pageid, 'format': 'json', 'explaintext': '1'}
            response = await self._make_api_request_async(lang, params)
            response.raise_for_status()
            data = response.json()
            pages = data.get('query', {}).get('pages', {})
            page = pages.get(str(pageid), {})
            if not page or 'missing' in page:
                raise Exception("Article not found")
            extract = page.get('extract', '')
            result = {'title': page.get('title', ''), 'content': extract, 'pageid': pageid}
            self._cache_article(pageid, result)
            return result
        
        lang = self._determine_language(language, title)
        
        parsed_sections = None
        if sections:
            parsed_sections = [s.strip() for s in sections.split(',')]
        
        cache_key = f"content:{lang}:{title}:{sections or 'full'}"
        current_time = datetime.now().timestamp()
        if (cache_key in self.cache and 
            current_time - self.cache_timestamp.get(cache_key, 0) < self.cache_expiry):
            print(f"Using cached content for {title}")
            return self.cache[cache_key]
        
        # First try the exact title
        try:
            content_data = await self._get_page_extract_async(title, lang, intro_only=False)
            
            # Extract and format sections if requested
            if parsed_sections:
                content_data = self._filter_sections(content_data, parsed_sections)
            
            # Cache and return
            self.cache[cache_key] = content_data
            self.cache_timestamp[cache_key] = current_time
            
            return content_data
            
        except Exception as direct_error:
            print(f"Direct title lookup failed: {str(direct_error)}")
            
            # If direct lookup fails, try searching
            try:
                search_results = await self.search_wikipedia(title, lang, limit=1)
                
                # Handle both list and dict response formats
                if isinstance(search_results, list) and len(search_results) > 0:
                    first_result = search_results[0]
                    actual_title = first_result['title']
                elif isinstance(search_results, dict) and search_results.get('count', 0) > 0:
                    first_result = search_results['results'][0]
                    actual_title = first_result['title']
                else:
                    raise ValueError(f"No Wikipedia article found for '{title}'")
                
                print(f"Using search result: {actual_title}")
                
                # Get the content using the found title
                content_data = await self._get_page_extract_async(actual_title, lang, intro_only=False)
                
                # Extract and format sections if requested
                if parsed_sections:
                    content_data = self._filter_sections(content_data, parsed_sections)
                
                # Cache the result
                self.cache[cache_key] = content_data
                self.cache_timestamp[cache_key] = current_time
                
                return content_data
                
            except Exception as search_error:
                error_msg = f"Error getting Wikipedia content: {str(search_error)}"
                print(error_msg)
                raise Exception(error_msg)
    
    async def _get_page_extract_async(self, title: str, language: str, intro_only: bool = False) -> Dict[str, Any]:
        """
        Get the extract (text content) of a Wikipedia page asynchronously.
        
        Args:
            title (str): Article title
            language (str): Wikipedia language code
            intro_only (bool, optional): Only get the introduction section (default: False)
        
        Returns:
            Dict[str, Any]: Article data with extract
            
        Raises:
            Exception: If the API request fails or article not found
        """
        # Prepare request parameters
        params = {
            'action': 'query',
            'prop': 'extracts|info|categories|images|pageimages|revisions|pageprops',
            'exintro': '1' if intro_only else '0',
            'explaintext': '1',
            'titles': title,
            'format': 'json',
            'inprop': 'url|displaytitle',
            'piprop': 'thumbnail',
            'pithumbsize': 300,
            'rvprop': 'timestamp',
            'rvlimit': 1,
            'redirects': '1',
            'cllimit': 5
        }
        
        data = await self._make_api_request_async(language, params)

        pages = data.get('query', {}).get('pages', {})
        
        if '-1' in pages and 'missing' in pages['-1']:
            raise Exception(f"Wikipedia article '{title}' not found")
        
        # Get the first page (should be only one)
        page_id = next(iter(pages.keys()))
        page = pages[page_id]
        
        page_title = page.get('title', title)
        page_url = page.get('fullurl', f"https://{language}.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}")
        extract = page.get('extract', '')
        
        revision = page.get('revisions', [{}])[0]
        timestamp = revision.get('timestamp', datetime.now().isoformat())
        
        thumbnail = None
        if 'thumbnail' in page:
            thumbnail = page['thumbnail'].get('source', None)
        
        categories = []
        if 'categories' in page:
            categories = [cat.get('title', '').replace('Category:', '') for cat in page.get('categories', [])]
        
        sections = self._extract_sections(extract)
        
        result = {
            'title': page_title,
            'pageid': int(page_id),
            'url': page_url,
            'language': language,
            'extract': extract,
            'thumbnail': thumbnail,
            'sections': sections,
            'categories': categories,
            'last_updated': timestamp,
            'timestamp': datetime.now().isoformat(),
            'attribution': f"Content from Wikipedia, retrieved {datetime.now().strftime('%Y-%m-%d')}"
        }
        
        return result
    
    def _extract_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract section titles and content from Wikipedia text.
        
        Args:
            text (str): Wikipedia article text
            
        Returns:
            List[Dict[str, Any]]: List of sections with titles and content
        """
        lines = text.split('\n')
        
        sections = []
        current_section = {'title': 'Introduction', 'level': 0, 'content': ''}
        
        for line in lines:
            header_match = re.match(r'^(=+)\s*(.+?)\s*\1$', line)
            
            if header_match:
                if current_section['content'].strip():
                    sections.append(current_section.copy())
                
                level = len(header_match.group(1))
                title = header_match.group(2)
                current_section = {
                    'title': title,
                    'level': level,
                    'content': ''
                }
            else:
                if current_section['content'] and line.strip():
                    current_section['content'] += '\n'
                current_section['content'] += line
        
        if current_section['content'].strip():
            sections.append(current_section)
        
        return sections
    
    def _filter_sections(self, article_data: Dict[str, Any], section_names: List[str]) -> Dict[str, Any]:
        """
        Filter article data to include only specified sections.
        
        Args:
            article_data (Dict[str, Any]): Full article data
            section_names (List[str]): List of section titles to include
            
        Returns:
            Dict[str, Any]: Filtered article data
        """
        filtered_data = article_data.copy()
        
        all_sections = article_data.get('sections', [])
        
        filtered_sections = [s for s in all_sections if s['title'] == 'Introduction']
        
        section_names_lower = [name.lower() for name in section_names]
        
        for section in all_sections:
            if section['title'].lower() in section_names_lower:
                filtered_sections.append(section)
        
        filtered_extract = ""
        for section in filtered_sections:
            if section['title'] == 'Introduction':
                filtered_extract += section['content']
            else:
                header = '=' * section['level']
                filtered_extract += f"\n\n{header} {section['title']} {header}\n{section['content']}"
        
        filtered_data['extract'] = filtered_extract
        filtered_data['sections'] = filtered_sections
        filtered_data['filtered_sections'] = True
        
        return filtered_data
    
    def _determine_language(self, language: str, text: str) -> str:
        """
        Determine the Wikipedia language to use.
        
        Args:
            language (str): User-specified language code or None
            text (str): Input text to analyze if language is not specified
            
        Returns:
            str: Language code to use
        """
        if language:
            return language.lower()
            
        # Detect Russian/Cyrillic text and use Russian Wikipedia
        if re.search(r'[а-яА-Я]', text):
            return 'ru'
            
        return self.default_language
    
    def _clean_html(self, text: str) -> str:
        """
        Clean HTML tags and entities from text.
        
        Args:
            text (str): Text with HTML
            
        Returns:
            str: Cleaned text
        """
        text = re.sub(r'<[^>]+>', ' ', text)
        
        text = html.unescape(text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_search_description(self, search_data: Dict[str, Any]) -> str:
        """
        Generate a human-readable description of Wikipedia search results.
        
        Args:
            search_data (Dict[str, Any]): Search data from search_wikipedia
            
        Returns:
            str: Human-readable search results
        """
        query = search_data['query']
        lang = search_data['language']
        count = search_data['count']
        results = search_data['results']
        
        if count == 0:
            return f"No Wikipedia articles found for '{query}' in {lang} Wikipedia."
        
        description = f"Found {count} Wikipedia articles for '{query}':\n\n"
        
        for i, result in enumerate(results):
            title = result['title']
            snippet = result['snippet']
            url = f"https://{lang}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
            
            description += f"{i+1}. {title}. "
            description += f"{snippet}. "
            description += f"URL: {url}.\n\n"
        
        description += f"Source: {lang}.wikipedia.org."
        
        return description
    
    def get_article_description(self, article_data: Dict[str, Any], is_summary: bool = False) -> str:
        """
        Generate a human-readable description of a Wikipedia article.
        
        Args:
            article_data (Dict[str, Any]): Article data from get_article_summary or get_article_content
            is_summary (bool, optional): Whether this is a summary (default: False)
            
        Returns:
            str: Article summary in natural language
        """
        title = article_data['title']
        url = article_data['url']
        extract = article_data['extract']
        attribution = article_data['attribution']
        
        if is_summary:
            heading = f"Wikipedia Summary: {title}.\n\n"
        else:
            heading = f"Wikipedia Article: {title}.\n\n"
        
        content = extract
        
        footer = f"\n\nSource: {url}.\n{attribution}"
        
        return heading + content + footer

    def set_language(self, lang_code: str):
        """Set the language for Wikipedia API requests."""
        import re
        if not isinstance(lang_code, str) or not re.match(r'^[a-z]{2}$', lang_code):
            raise ValueError("Invalid language code")
        self.language = lang_code

    def _get_api_url(self) -> str:
        """Return the formatted API URL for the current language."""
        return self.api_base_url.format(lang=self.language)

    def _cache_article(self, pageid: int, data: Dict[str, Any]) -> None:
        """Cache article content by pageid."""
        key = f"article:{pageid}"
        self.cache[key] = data
        self.cache_timestamp[key] = datetime.now().timestamp()

    def _get_cached_article(self, pageid: int) -> Optional[Dict[str, Any]]:
        """Retrieve cached article content by pageid."""
        key = f"article:{pageid}"
        if key in self.cache and datetime.now().timestamp() - self.cache_timestamp.get(key, 0) < self.cache_expiry:
            return self.cache[key]
        return None

# Functions to expose to the LLM tool system
@cached(wiki_cache)
async def search_wikipedia_async(query: str, limit: int = 5) -> str:
    """Searches Wikipedia asynchronously for pages matching the query."""
    print(f"search_wikipedia_async called for '{query}' (cache miss or expired)")
    tool = WikipediaTool()
    try:
        search_results = await tool.search_wikipedia(query, limit=int(limit))
        return tool.format_search_results(search_results, query)
    except Exception as e:
        return f"Error searching Wikipedia for '{query}': {str(e)}"

@cached(wiki_cache)
async def get_article_summary_async(title: str, sentences: int = 3) -> str:
    """Gets a summary of a specific Wikipedia article asynchronously."""
    print(f"get_article_summary_async called for '{title}' (cache miss or expired)")
    tool = WikipediaTool()
    try:
        summary = await tool.get_article_summary(title)
        return tool.format_summary(summary, title)
    except Exception as e:
        return f"Error getting summary for Wikipedia article '{title}': {str(e)}"

@cached(wiki_cache)
async def get_article_content_async(title: str, section: Optional[str] = None) -> str:
    """Gets the full content or a specific section of a Wikipedia article asynchronously."""
    print(f"get_article_content_async called for '{title}' (section: {section}) (cache miss or expired)")
    tool = WikipediaTool()
    try:
        content = await tool.get_article_content(title, section)
        return tool.format_content(content, title, section)
    except Exception as e:
        return f"Error getting content for Wikipedia article '{title}' (section: {section}): {str(e)}"

# --- Tool Definitions using Synchronous Wrappers --- 

_wiki_tool_instance = WikipediaTool() # Create a single instance

@tool(
    name="wikipedia_search",
    description="Searches Wikipedia for articles matching a query and returns a list of titles, snippets, and page IDs.",
    parameters={
        "query": {"type": "string", "description": "The search term or query.", "required": True},
        "language": {"type": "string", "description": "Wikipedia language code (e.g., 'en', 'es', 'de'). Defaults to 'en'.", "optional": True},
        "limit": {"type": "integer", "description": "Maximum number of search results to return.", "default": 5}
    }
)
def wikipedia_search_tool(query: str, language: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    """Synchronous wrapper for WikipediaTool.search_wikipedia"""
    try:
        return asyncio.run(_wiki_tool_instance.search_wikipedia(query=query, language=language, limit=limit))
    except Exception as e:
        # Log the error appropriately if logging is set up here
        print(f"Error in wikipedia_search_tool: {e}")
        # Reraise or return an error structure suitable for the ToolManager
        raise e 

@tool(
    name="wikipedia_get_summary",
    description="Fetches a concise summary (usually the introduction) of a specific Wikipedia article.",
    parameters={
        "title": {"type": "string", "description": "The exact title of the Wikipedia article (case-sensitive).", "required": True},
        "language": {"type": "string", "description": "Wikipedia language code (e.g., 'en', 'es', 'de'). Defaults to 'en'.", "optional": True}
    }
)
def wikipedia_summary_tool(title: str, language: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous wrapper for WikipediaTool.get_article_summary"""
    try:
        # Returns a Dict like {'title': ..., 'summary': ..., 'url': ..., 'pageid': ...}
        return asyncio.run(_wiki_tool_instance.get_article_summary(title=title, language=language))
    except Exception as e:
        print(f"Error in wikipedia_summary_tool: {e}")
        raise e

@tool(
    name="wikipedia_get_content",
    description="Retrieves the text content of a Wikipedia article, optionally filtering by specific section titles.",
    parameters={
        "title": {"type": "string", "description": "The exact title of the Wikipedia article (case-sensitive).", "required": True},
        "language": {"type": "string", "description": "Wikipedia language code (e.g., 'en', 'es', 'de'). Defaults to 'en'.", "optional": True},
        "sections": {"type": "string", "description": "Comma-separated list of exact section titles to retrieve (optional). If omitted, retrieves full content.", "optional": True}
    }
)
def wikipedia_content_tool(title: str, language: Optional[str] = None, sections: Optional[str] = None) -> Dict[str, Any]:
    """Synchronous wrapper for WikipediaTool.get_article_content"""
    try:
        # Returns a Dict like {'title': ..., 'content': ..., 'url': ..., 'pageid': ..., ('sections': ... if requested)}
        return asyncio.run(_wiki_tool_instance.get_article_content(title=title, language=language, sections=sections))
    except Exception as e:
        print(f"Error in wikipedia_content_tool: {e}")
        raise e