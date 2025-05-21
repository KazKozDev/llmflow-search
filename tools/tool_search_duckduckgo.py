#!/usr/bin/env python3
"""
LLMFlow Search Agent - DuckDuckGo Search Tool
Implementation of DuckDuckGo search functionality with advanced features.
Includes caching, proxy support, and Selenium integration for better results.
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sys
import time
import urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import requests
from bs4 import BeautifulSoup

# Selenium imports
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# Configure logging
logger = logging.getLogger("duckduckgo_searcher")

# Constants
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:96.0) Gecko/20100101 Firefox/96.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36",
]

# Create cache dir in tools directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
CACHE_EXPIRATION = timedelta(hours=24)

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Created cache directory at {CACHE_DIR}")
    except Exception as e:
        logger.error(f"Failed to create cache directory: {e}")

# Definition of the exported variable
default_searcher = None

class DuckDuckGoSearcher:
    """A class to search DuckDuckGo and extract results, optionally using Selenium."""

    def __init__(
        self,
        use_cache: bool = True,
        use_proxy: bool = False,
        max_retries: int = 3,
        verbose: bool = False,
        use_selenium: bool = True,
        chromedriver_path: Optional[str] = None
    ) -> None:
        """Initialize the searcher."""
        self.use_cache = use_cache
        self.use_proxy = use_proxy
        self.max_retries = max_retries
        self.verbose = verbose
        self.use_selenium = use_selenium
        self.chromedriver_path = chromedriver_path

        if use_cache and not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

        logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    def _get_selenium_driver(self) -> Optional[webdriver.Chrome]:
        """Initializes and returns a Selenium WebDriver instance."""
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument(f"--user-agent={self.get_random_user_agent()}")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)

        try:
            if self.chromedriver_path:
                service = ChromeService(executable_path=self.chromedriver_path)
                driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                driver = webdriver.Chrome(options=chrome_options)
            
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            return driver
        except WebDriverException as e:
            logger.error(f"Failed to initialize Selenium WebDriver: {e}")
            logger.error("Please ensure ChromeDriver is installed and in your PATH, or specify chromedriver_path.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during WebDriver initialization: {e}")
            return None

    def _fetch_page_with_selenium(self, url: str) -> Optional[str]:
        """Fetches page content using Selenium."""
        driver = self._get_selenium_driver()
        if not driver:
            return None
        
        try:
            logger.debug(f"Fetching URL with Selenium: {url}")
            driver.get(url)
            time.sleep(random.uniform(2, 4))
            html_content = driver.page_source
            if self.verbose:
                logger.debug(f"Page content fetched with Selenium (length: {len(html_content)})")
            return html_content
        except TimeoutException:
            logger.warning(f"Timeout while loading page with Selenium: {url}")
            return None
        except WebDriverException as e:
            logger.warning(f"WebDriverException while fetching page {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching page {url} with Selenium: {e}")
            return None
        finally:
            if driver:
                driver.quit()

    def get_random_user_agent(self) -> str:
        """Return a random User-Agent."""
        return random.choice(USER_AGENTS)

    def get_proxies(self) -> Optional[Dict[str, str]]:
        """Get a random proxy (placeholder)."""
        if not self.use_proxy:
            return None

        proxy_list = [
            "http://proxy1.example.com:8080",
            "http://proxy2.example.com:8080",
        ]

        if not proxy_list or all("example.com" in proxy for proxy in proxy_list):
            logger.warning("No valid proxies found. Using direct connection.")
            return None

        proxy_address = random.choice(proxy_list)
        proxies = {"http": proxy_address, "https": proxy_address}
        logger.info(f"Using proxy: {proxy_address}")
        return proxies

    def get_cache_path(self, query: str) -> str:
        """Get the cache file path for a query."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return os.path.join(CACHE_DIR, f"{query_hash}.json")

    def get_cached_results(self, query: str) -> Optional[List[Dict]]:
        """Get cached results if not expired."""
        if not self.use_cache:
            return None

        cache_path = self.get_cache_path(query)
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            cached_time = datetime.fromisoformat(cached_data["timestamp"])
            if datetime.now() - cached_time > CACHE_EXPIRATION:
                logger.debug(f"Cache for '{query}' has expired.")
                return None

            logger.info(f"Using cached results for '{query}'")
            return cached_data["results"]
        except Exception as e:
            logger.warning(f"Error reading cache: {e}")
            return None

    def save_to_cache(self, query: str, results: List[Dict]) -> None:
        """Save search results to cache."""
        if not self.use_cache or not results:
            return

        cache_path = self.get_cache_path(query)
        try:
            cache_data = {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Results for '{query}' saved to cache")
        except Exception as e:
            logger.warning(f"Error saving to cache: {e}")

    def search(self, query: str) -> List[Dict]:
        """Search DuckDuckGo for the given query."""
        cached_results = self.get_cached_results(query)
        if cached_results is not None:
            return cached_results

        logger.info(f"Searching DuckDuckGo for: {query} {'(using Selenium)' if self.use_selenium else ''}")
        results = self._search_html_version(query)

        if not results:
            logger.info(f"HTML version {'failed or no results' if self.use_selenium else 'failed'}, trying lite version {'(Selenium may not be effective here)' if self.use_selenium else ''}")
            results = self._search_lite_version(query)

        if results:
            self.save_to_cache(query, results)
        else:
            logger.warning(f"No results found for query: {query} after trying all methods.")

        return results

    def _search_html_version(self, query: str) -> List[Dict]:
        """Search using the HTML version of DuckDuckGo."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://duckduckgo.com/html/?q={encoded_query}"
        
        html_content = None
        if self.use_selenium:
            html_content = self._fetch_page_with_selenium(url)
        else:
            response = self._make_request_requests(url)
            if response:
                html_content = response.text

        return self._extract_html_results(html_content) if html_content else []

    def _search_lite_version(self, query: str) -> List[Dict]:
        """Search using the lite version of DuckDuckGo."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://lite.duckduckgo.com/lite/?q={encoded_query}"
        html_content = None
        if self.use_selenium:
            html_content = self._fetch_page_with_selenium(url)
        else:
            response = self._make_request_requests(url)
            if response:
                html_content = response.text

        return self._extract_lite_results(html_content) if html_content else []

    def _make_request_requests(self, url: str) -> Optional[requests.Response]:
        """Make an HTTP request with retries using `requests` library."""
        retry_count = 0
        proxies = None
        
        max_retries_requests = 5

        while retry_count < max_retries_requests:
            try:
                time.sleep(random.uniform(0.5, 1.5))
                user_agent = self.get_random_user_agent()
                headers = {
                    "User-Agent": user_agent,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
                    "Referer": "https://duckduckgo.com/",
                }
                
                response = requests.get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout=15,
                )
                if response.status_code == 200:
                    if any(
                        term in response.text.lower()
                        for term in ["captcha", "blocked", "too many requests"]
                    ):
                        logger.warning(f"CAPTCHA or blocking detected by requests on {url}. Retrying...")
                        retry_count += 1
                        time.sleep(2**retry_count + random.uniform(0.5, 1.5))
                        continue
                    return response
                elif response.status_code == 429 or response.status_code >= 500:
                    logger.warning(f"Requests: Got status code {response.status_code} for {url}. Retrying...")
                    retry_count += 1
                    time.sleep(2**retry_count + random.uniform(0.5, 1.5))
                else:
                    logger.error(f"Requests: Error Got status code {response.status_code} for {url}")
                    if self.verbose and response.status_code not in [429, 500, 502, 503, 504]:
                        debug_path = f"debug_response_{response.status_code}.html"
                        try:
                            with open(debug_path, "w", encoding="utf-8") as f:
                                f.write(response.text)
                            logger.debug(f"Saved error response from {url} to {debug_path}")
                        except Exception as e_save:
                            logger.error(f"Failed to save debug response: {e_save}")
                    return None
            except requests.exceptions.RequestException as e:
                logger.warning(f"Requests: Request error for {url}: {e}")
                retry_count += 1
                time.sleep(2**retry_count + random.uniform(0.5, 1.5))
            except Exception as e:
                logger.error(f"Requests: Unexpected error for {url}: {e}")
                return None
        logger.error(f"Requests: Failed to make request to {url} after {max_retries_requests} retries")
        return None

    def _extract_html_results(self, html_content: Optional[str]) -> List[Dict]:
        """Extract search results from HTML version."""
        if not html_content:
            return []
        results = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            if self.verbose:
                debug_file = "debug_ddg_html_selenium.html" if self.use_selenium else "debug_ddg_html_requests.html"
                try:
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    logger.debug(f"Saved DDG HTML content to {debug_file}")
                except Exception as e_save:
                    logger.error(f"Failed to save DDG HTML content: {e_save}")

            selectors_to_try = [
                "div.result", "div.results_links_deep", "div.web-result",
                "article.result", "li[data-layout=\"organic\"]", "div[data-testid=\"result\"]"
            ]
            result_elements = []
            for selector in selectors_to_try:
                result_elements = soup.select(selector)
                if result_elements:
                    logger.debug(f"Found {len(result_elements)} result elements using selector: {selector}")
                    break
            else:
                logger.warning("No result elements found with any of the tried selectors (HTML version).")
                return []

            for result_container in result_elements:
                title, link, snippet = "", "", ""
                title_el = result_container.select_one("h2 a, a.result__a, a[data-testid='result-title-a']")
                snippet_el = result_container.select_one(".result__snippet, a.result__snippet, div[data-testid='result-snippet']")
                link_el = title_el

                if title_el:
                    title = title_el.get_text(strip=True)
                if link_el:
                    raw_link = link_el.get("href")
                    if raw_link:
                        if raw_link.startswith("/ κάποιο redirect") or "duckduckgo.com/y.js" in raw_link:
                            parsed_url = urllib.parse.urlparse(raw_link)
                            qs = urllib.parse.parse_qs(parsed_url.query)
                            link = qs.get("uddg", [""])[0] or qs.get("u",[""])[0] 
                        else:
                            link = raw_link
                
                if snippet_el:
                    snippet = snippet_el.get_text(strip=True)
                
                if title and link:
                    results.append({"title": title, "link": link, "content": snippet, "source": "DuckDuckGo"})
                elif self.verbose:
                    logger.debug(f"Skipped a result container, couldn't extract title/link. Container: {result_container.prettify()[:200]}...")

        except Exception as e:
            logger.error(f"Error extracting HTML results: {e}")
            if self.verbose:
                logger.debug(f"Problematic HTML content snippet: {html_content[:1000]}")
        return results

    def _extract_lite_results(self, html_content: Optional[str]) -> List[Dict]:
        """Extract search results from the lite version of DuckDuckGo."""
        if not html_content:
            return []
        results = []
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            if self.verbose:
                debug_file = "debug_ddg_lite_selenium.html" if self.use_selenium else "debug_ddg_lite_requests.html"
                try:
                    with open(debug_file, "w", encoding="utf-8") as f:
                        f.write(html_content)
                    logger.debug(f"Saved DDG Lite HTML content to {debug_file}")
                except Exception as e_save:
                    logger.error(f"Failed to save DDG Lite HTML: {e_save}")

            links = soup.find_all('a', href=True)
            current_title = ""
            current_link = ""
            current_snippet = ""

            rows = soup.select('tr')
            for i, row in enumerate(rows):
                link_tag = row.find('a', href=True)
                if link_tag and link_tag.get_text(strip=True):
                    title = link_tag.get_text(strip=True)
                    link = link_tag['href']
                    if link.startswith("/") or "duckduckgo.com" in link or "ad_domain" in link_tag.get('class',[]):
                        continue
                    
                    snippet = ""
                    non_link_text_in_row = ''.join(s for s in row.find_all(string=True, recursive=False) if s.strip()) 
                    snippet_parts_in_row = [td.get_text(strip=True) for td in row.find_all('td') if not td.find('a')] 
                    if snippet_parts_in_row:
                        snippet = " ".join(snippet_parts_in_row).strip()
                    elif non_link_text_in_row:
                         snippet = non_link_text_in_row

                    if title and link:
                        results.append({"title": title, "link": link, "content": snippet, "source": "DuckDuckGo (Lite)"})
            
            if not results and self.verbose:
                logger.warning("No results extracted from Lite version using current selectors.")

        except Exception as e:
            logger.error(f"Error extracting Lite results: {e}")
            if self.verbose:
                logger.debug(f"Problematic Lite HTML content snippet: {html_content[:1000]}")
        return results

def display_results(results: List[Dict], colorize: bool = True) -> None:
    """Display search results in the console."""
    if not results:
        print("No results found or an error occurred.")
        return

    print(f"\nFound {len(results)} results:\n")
    colors = {
        "title": "\033[1;36m",
        "link": "\033[0;32m",
        "desc": "\033[0;37m",
        "reset": "\033[0m",
    }

    if not colorize or (os.name == "nt" and not os.environ.get("ANSICON")):
        colors = {k: "" for k in colors}

    for i, result in enumerate(results, 1):
        print(f"{i}. {colors['title']}{result['title']}{colors['reset']}")
        print(f"   {colors['link']}{result['link']}{colors['reset']}")
        print(f"   {colors['desc']}{result['content']}{colors['reset']}")
        print()

def save_results_to_file(results: List[Dict], filename: str) -> bool:
    """Save search results to a text file."""
    if not results:
        return False

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"DuckDuckGo Search Results ({len(results)} items)\n")
            f.write("=" * 50 + "\n\n")
            for i, result in enumerate(results, 1):
                f.write(f"{i}. {result['title']}\n")
                f.write(f"   {result['link']}\n")
                f.write(f"   {result['content']}\n\n")
        print(f"Results saved to {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False

def save_results_to_json(results: List[Dict], filename: str) -> bool:
    """Save search results to a JSON file."""
    if not results:
        return False

    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to JSON format: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving JSON: {e}")
        return False

# Tool function for ToolManager integration
try:
    from .tool_manager import tool
except ImportError:
    # Fallback if run standalone or import fails
    def tool(name=None, description=None, parameters=None):
        def decorator(func):
            return func
        return decorator

_default_ddg_searcher = DuckDuckGoSearcher(use_selenium=True, verbose=False) # Configure as needed, maybe read from env/config

@tool(
    name="duckduckgo_search",
    description="Performs a DuckDuckGo search and returns a list of results (title, link, snippet). Uses Selenium by default.",
    parameters={
        "query": {"type": "string", "description": "The search query string.", "required": True},
        "limit": {"type": "integer", "description": "Maximum number of results to return.", "default": 10, "required": False}
    }
)
def duckduckgo_search_tool(query: str, limit: int = 10) -> List[Dict]:
    """
    Performs a DuckDuckGo search using the DuckDuckGoSearcher class.
    
    Args:
        query: The search term.
        limit: Maximum number of results.
        
    Returns:
        A list of search result dictionaries.
    """
    logger.info(f"Executing DuckDuckGo search tool for query: '{query}' with limit {limit}")
    try:
        # Consider if the searcher needs specific options per call, or if a default is fine
        # For now, using the default instance initialized above.
        results = _default_ddg_searcher.search(query)
        if limit > 0 and len(results) > limit:
            results = results[:limit]
        logger.info(f"DuckDuckGo search tool found {len(results)} results.")
        return results
    except Exception as e:
        logger.error(f"Error during duckduckgo_search_tool execution: {e}", exc_info=True)
        return [{"error": f"Failed to execute DuckDuckGo search: {str(e)}"}]

def main() -> None:
    """Main function for the script."""
    parser = argparse.ArgumentParser(description="DuckDuckGo Search without Blocking")
    parser.add_argument("query", nargs="*", help="Search query (prompted if not provided)")
    parser.add_argument("--proxy", "-p", action="store_true", help="Use proxy rotation (requires setup)")
    parser.add_argument("--no-cache", "-n", action="store_true", help="Disable caching of results")
    parser.add_argument("--retries", "-r", type=int, default=5, help="Maximum number of retries")
    parser.add_argument("--output", "-o", help="Save results to a text file")
    parser.add_argument("--json", "-j", help="Save results to a JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed information")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit the number of results")
    parser.add_argument("--use-selenium", action="store_true", help="Use Selenium for fetching")
    parser.add_argument("--chromedriver-path", type=str, default=None, help="Path to chromedriver executable")
    args = parser.parse_args()

    query = " ".join(args.query) if args.query else input("Enter search query: ")
    print(f"Performing DuckDuckGo search: {query}")
    print("Please wait...")

    searcher = DuckDuckGoSearcher(
        use_cache=not args.no_cache,
        use_proxy=args.proxy,
        max_retries=args.retries,
        verbose=args.verbose,
        use_selenium=args.use_selenium,
        chromedriver_path=args.chromedriver_path
    )

    results = searcher.search(query)
    if args.limit > 0 and len(results) > args.limit:
        results = results[:args.limit]

    if results:
        display_results(results)
        if args.output:
            save_results_to_file(results, args.output)
        if args.json:
            save_results_to_json(results, args.json)
    else:
        print("Search failed or no results found.")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
else:
    # Initialization of default_searcher for import from other modules
    if default_searcher is None:
        default_searcher = DuckDuckGoSearcher(use_cache=True, verbose=False)