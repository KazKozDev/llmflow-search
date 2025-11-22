#!/usr/bin/env python3
"""
LLMFlow Search Agent - Async Link Parsing Tool
Async implementation of web content extraction with improved performance.
"""

import aiohttp
import asyncio
from bs4 import BeautifulSoup
from newspaper import Article
from urllib.parse import urlparse
import re
import logging
from readability import Document
from typing import Tuple, Optional

# Setup logger
logger = logging.getLogger(__name__)

# Modern User-Agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

def is_valid_url(url: str) -> bool:
    """Check if the given string is a valid URL."""
    if not url:
        return False
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.error(f"Error validating URL {url}: {e}")
        return False


async def method1_bs4_async(url: str, session: aiohttp.ClientSession) -> str:
    """Parse main content from URL using BeautifulSoup (async)."""
    logger.debug(f"Method 1 (BeautifulSoup) attempting: {url}")
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=7)) as response:
            response.raise_for_status()
            html = await response.text()
        
        # Parse in executor (CPU bound)
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _parse_bs4, html, url)
        return content
        
    except asyncio.TimeoutError:
        logger.warning(f"Timeout in method1_bs4 for {url}")
        return "Error: Timeout"
    except Exception as e:
        logger.error(f"Error in method1_bs4 for {url}: {e}")
        return f"Error: {str(e)}"


def _parse_bs4(html: str, url: str) -> str:
    """CPU-bound BeautifulSoup parsing."""
    soup = BeautifulSoup(html, 'html.parser')
    
    # Remove non-content elements
    for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button', 'input']):
        tag.decompose()
    
    article_content = ""
    
    # Special handling for cryptocurrency pages
    if any(keyword in url.lower() for keyword in ['bitcoin', 'btc', 'crypto', 'price', 'prediction']):
        price_data = []
        tables = soup.find_all('table')
        for table in tables:
            table_data = ""
            headers = [th.get_text(strip=True) for th in table.find_all('th')]
            if headers:
                table_data += " | ".join(headers) + "\n"
            
            for row in table.find_all('tr'):
                row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                if row_data:
                    table_data += " | ".join(row_data) + "\n"
            
            if table_data.strip():
                price_data.append(table_data.strip())
        
        if price_data:
            article_content += "PRICE PREDICTION TABLES:\n" + "\n\n".join(price_data) + "\n\n"
        
        if article_content.strip():
            return article_content.strip()
    
    # Standard extraction - try article tag
    article_tag = soup.find('article')
    if article_tag:
        for p in article_tag.find_all('p'):
            article_content += p.get_text(separator='\n', strip=True) + "\n\n"
        if article_content.strip():
            return article_content.strip()
    
    # Content divs
    content_divs = soup.find_all('div', class_=lambda c: c and any(
        key in c.lower() for key in ['content', 'article', 'main', 'body', 'post', 'entry']
    ))
    for div in content_divs:
        for p in div.find_all('p'):
            article_content += p.get_text(separator='\n', strip=True) + "\n\n"
        if article_content.strip():
            return article_content.strip()
    
    # Fallback: all paragraphs
    if not article_content:
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            article_content += p.get_text(separator='\n', strip=True) + "\n\n"
    
    return article_content.strip()


async def method2_newspaper_async(url: str) -> str:
    """Parse content using Newspaper3k (runs in executor)."""
    logger.debug(f"Method 2 (Newspaper3k) attempting: {url}")
    try:
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _newspaper_parse, url)
        return content
    except Exception as e:
        logger.error(f"Error in method2_newspaper for {url}: {e}")
        return f"Error: {str(e)}"


def _newspaper_parse(url: str) -> str:
    """CPU-bound Newspaper parsing."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip() if article.text else ""
    except Exception as e:
        raise e


async def method3_readability_async(url: str, session: aiohttp.ClientSession) -> str:
    """Parse content using Readability (async)."""
    logger.debug(f"Method 3 (Readability) attempting: {url}")
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=7)) as response:
            response.raise_for_status()
            html = await response.text()
        
        # Parse in executor
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, _readability_parse, html)
        return content
        
    except asyncio.TimeoutError:
        logger.warning(f"Timeout in method3_readability for {url}")
        return "Error: Timeout"
    except Exception as e:
        logger.error(f"Error in method3_readability for {url}: {e}")
        return f"Error: {str(e)}"


def _readability_parse(html: str) -> str:
    """CPU-bound Readability parsing."""
    doc = Document(html)
    content_html = doc.summary()
    soup = BeautifulSoup(content_html, 'html.parser')
    clean_text = soup.get_text(separator='\n', strip=True)
    clean_text = re.sub(r'\n{2,}', '\n\n', clean_text)
    return clean_text.strip()


async def compare_methods_async(url: str) -> Tuple[str, str]:
    """
    Compare parsing methods with early exit optimization.
    Returns: (content, method_used)
    """
    logger.debug(f"Comparing parsing methods for {url}")
    
    # Create session with modern headers and SSL handling
    connector = aiohttp.TCPConnector(ssl=False)  # SSL fallback
    headers = {'User-Agent': USER_AGENT}
    
    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        # Try Readability first (usually best quality)
        readability_result = await method3_readability_async(url, session)
        if readability_result and not readability_result.startswith("Error") and len(readability_result) > 500:
            logger.info(f"Early exit: readability gave {len(readability_result)} chars for {url}")
            return readability_result, "readability"
        
        # Try Newspaper (good for articles)
        newspaper_result = await method2_newspaper_async(url)
        if newspaper_result and not newspaper_result.startswith("Error") and len(newspaper_result) > 500:
            logger.info(f"Early exit: newspaper gave {len(newspaper_result)} chars for {url}")
            return newspaper_result, "newspaper"
        
        # Try BeautifulSoup
        bs4_result = await method1_bs4_async(url, session)
        if bs4_result and not bs4_result.startswith("Error") and len(bs4_result) > 200:
            logger.info(f"Selected bs4: {len(bs4_result)} chars for {url}")
            return bs4_result, "bs4"
        
        # Return the longest non-error result
        results = {
            "readability": readability_result,
            "newspaper": newspaper_result,
            "bs4": bs4_result
        }
        
        best_result = ""
        best_method = "none"
        best_length = 0
        
        for method, result in results.items():
            if result and not result.startswith("Error") and len(result) > best_length:
                best_result = result
                best_method = method
                best_length = len(result)
        
        if best_result:
            logger.info(f"Selected {best_method} (longest): {best_length} chars for {url}")
            return best_result, best_method
        
        # All failed
        logger.warning(f"All methods failed for {url}")
        return readability_result or "Error: All methods failed", "failed"


def clean_text(text: str) -> str:
    """Clean extracted text from unwanted elements."""
    if not text or text.startswith("Error"):
        return text
    
    text = re.sub(r'\n{2,}', '\n\n', text.strip())
    
    # Remove common boilerplate patterns
    patterns_to_remove = [
        r"Subscribe to.*", r"Read also:.*", r"Share.*",
        r"Comments.*", r"Copyright ©.*", r"\d+ comments.*",
        r"Advertisement.*", r"Loading comments.*",
        r"Cookie Policy.*", r"Privacy Policy.*",
        r"Follow us on.*", r"Sign up for.*"
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Filter short lines
    lines = text.split('\n')
    meaningful_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if len(line_stripped) > 25 or '.' in line_stripped:
            meaningful_lines.append(line)
    
    text = '\n\n'.join(meaningful_lines).strip()
    return text


async def extract_content_from_url_async(url: str) -> str:
    """
    Main async function to extract content from URL.
    
    Args:
        url: URL to extract content from
        
    Returns:
        Extracted and cleaned text content
    """
    logger.info(f"Extracting content from: {url}")
    
    if not url or not is_valid_url(url):
        error_msg = f"Invalid URL: {url}"
        logger.error(error_msg)
        return f"Error: {error_msg}"
    
    try:
        # Get content using best method
        content, method_used = await compare_methods_async(url)
        original_length = len(content) if content else 0
        
        logger.info(f"Extracted {original_length} chars from {url} using {method_used}")
        
        if not content or content.startswith("Error"):
            return content or "Error: No content extracted"
        
        # Clean text
        cleaned_content = clean_text(content)
        cleaned_length = len(cleaned_content)
        
        logger.info(f"Cleaned content: {original_length} -> {cleaned_length} chars")
        
        # If cleaning removed too much, use lighter cleaning
        if cleaned_length < 200 and original_length > 1000:
            logger.warning("Cleaning too aggressive, using original")
            return re.sub(r'\n{3,}', '\n\n', content.strip())
        
        return cleaned_content if cleaned_content else "Error: Content empty after cleaning"
        
    except Exception as e:
        error_msg = f"Critical error extracting {url}: {str(e)}"
        logger.exception(error_msg)
        return f"Error: {error_msg}"


# Sync wrapper for backward compatibility
def extract_content_from_url(url: str, headers: dict = None) -> str:
    """
    Synchronous wrapper for extract_content_from_url_async.
    Kept for backward compatibility.
    """
    return asyncio.run(extract_content_from_url_async(url))
