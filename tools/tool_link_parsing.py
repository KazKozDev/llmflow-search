#!/usr/bin/env python3
"""
LLMFlow Search Agent - Link Parsing Tool
Implementation of web content extraction and link parsing functionality.
Extracts meaningful content from web pages for analysis.
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
from bs4 import BeautifulSoup
from newspaper import Article
from urllib.parse import urlparse
import re
import sys
from readability import Document
import logging
from functools import wraps
import os

# Setup a logger for this module
tool_link_parser_logger = logging.getLogger(__name__)
# Ensure the logger is configured (it should be by ToolManager, but good to have a fallback)
if not tool_link_parser_logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
    handler.setFormatter(formatter)
    tool_link_parser_logger.addHandler(handler)
    tool_link_parser_logger.setLevel(logging.INFO) # Set your desired level

# Definition of the 'tool' decorator (copied from tool_manager.py for robustness if import fails)
def tool(name: str = None, description: str = None, parameters: dict = None):
    """Decorator to register a function as a tool."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper._is_tool = True
        wrapper._tool_name = name or func.__name__
        wrapper._tool_description = description or func.__doc__ or "No description"
        wrapper._tool_parameters = parameters or {}
        return wrapper
    return decorator

def get_url_from_user():
    """Get URL from user input or stdin."""
    try:
        if not sys.stdin.isatty():
            url_data = sys.stdin.buffer.read()
            if isinstance(url_data, bytes):
                url_data = url_data.decode('utf-8', errors='replace')
            return url_data.strip()
        else:
            return input("Enter URL: ").strip() # Added prompt for interactive mode
    except Exception as e:
        tool_link_parser_logger.error(f"Error reading input: {e}")
        return ""

def is_valid_url(url):
    """Check if the given string is a valid URL."""
    if not url:
        return False
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False
    except Exception as e:
        tool_link_parser_logger.error(f"Error validating URL {url}: {e}")
        return False

def method1_bs4(url):
    """Parse main content from URL using BeautifulSoup."""
    tool_link_parser_logger.debug(f"Method 1 (BeautifulSoup) attempting: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove non-content elements
        for tag in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'aside', 'form', 'button', 'input']):
            tag.decompose()
        
        article_content = ""
        
        # Special handling for cryptocurrency price prediction pages
        if any(keyword in url.lower() for keyword in ['bitcoin', 'btc', 'crypto', 'price', 'prediction']):
            tool_link_parser_logger.debug(f"Detected cryptocurrency page, using specialized extraction for {url}")
            # Look for structured data tables and price information
            price_data = []
            
            # Extract tables which often contain price predictions
            tables = soup.find_all('table')
            for table in tables:
                table_data = ""
                # Get table headers
                headers = [th.get_text(strip=True) for th in table.find_all('th')]
                if headers:
                    table_data += " | ".join(headers) + "\n"
                
                # Get table rows
                for row in table.find_all('tr'):
                    row_data = [td.get_text(strip=True) for td in row.find_all('td')]
                    if row_data:
                        table_data += " | ".join(row_data) + "\n"
                
                if table_data.strip():
                    price_data.append(table_data.strip())
            
            # Look for price prediction paragraphs
            price_paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                # Filter for paragraphs containing price information
                if any(term in text.lower() for term in ['price', 'prediction', 'forecast', '$', '₿', 'btc', 'bitcoin']) and len(text) > 40:
                    price_paragraphs.append(text)
            
            # Add price tables and relevant paragraphs
            if price_data:
                article_content += "PRICE PREDICTION TABLES:\n" + "\n\n".join(price_data) + "\n\n"
            if price_paragraphs:
                article_content += "PRICE PREDICTION INFORMATION:\n" + "\n\n".join(price_paragraphs) + "\n\n"
            
            # If we found relevant content, return it
            if article_content.strip():
                return article_content.strip()
        
        # Standard content extraction - check for article tags first
        article_tag = soup.find('article')
        if article_tag:
            for p in article_tag.find_all('p'): 
                article_content += p.get_text(separator='\\n', strip=True) + "\\n\\n"
            if article_content.strip(): 
                return article_content.strip()
        
        # Look for content in divs with relevant class names
        content_divs = soup.find_all('div', class_=lambda c: c and any(key in c.lower() for key in ['content', 'article', 'main', 'body', 'post', 'entry', 'blog']))
        for div in content_divs:
            for p in div.find_all('p'): 
                article_content += p.get_text(separator='\\n', strip=True) + "\\n\\n"
            if article_content.strip(): 
                return article_content.strip()
        
        # If no specific content areas found, try all paragraphs
        if not article_content:
            paragraphs = soup.find_all('p')
            for p in paragraphs: 
                article_content += p.get_text(separator='\\n', strip=True) + "\\n\\n"
        
        return article_content.strip()
    except Exception as e:
        tool_link_parser_logger.error(f"Error in method1_bs4 for {url}: {e}")
        return f"Error in method 1 (BS4): {str(e)}"

def method2_newspaper(url):
    """Parse main content from URL using Newspaper3k."""
    tool_link_parser_logger.debug(f"Method 2 (Newspaper3k) attempting: {url}")
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip() if article.text else ""
    except Exception as e:
        tool_link_parser_logger.error(f"Error in method2_newspaper for {url}: {e}")
        return f"Error in method 2 (Newspaper3k): {str(e)}"

def method3_readability(url):
    """Parse main content from URL using Readability."""
    tool_link_parser_logger.debug(f"Method 3 (Readability) attempting: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        doc = Document(response.text)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, 'html.parser')
        clean_text_content = soup.get_text(separator='\\n', strip=True)
        clean_text_content = re.sub(r'\\n{2,}', '\\n\\n', clean_text_content) # Consolidate multiple newlines
        return clean_text_content.strip()
    except Exception as e:
        tool_link_parser_logger.error(f"Error in method3_readability for {url}: {e}")
        return f"Error in method 3 (Readability): {str(e)}"

def method4_direct_extraction(url):
    """Direct extraction of text from all elements."""
    tool_link_parser_logger.debug(f"Method 4 (Direct Extraction) attempting: {url}")
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'aside', 'form', 'button', 'input']):
            tag.decompose()
        all_text = soup.get_text(separator='\\n', strip=True)
        clean_text_content = re.sub(r'\\n{2,}', '\\n\\n', all_text) # Consolidate multiple newlines
        return clean_text_content.strip()
    except Exception as e:
        tool_link_parser_logger.error(f"Error in method4_direct_extraction for {url}: {e}")
        return f"Error in method 4 (Direct Extraction): {str(e)}"

def compare_methods(url):
    """Compare different parsing methods and select the best result."""
    tool_link_parser_logger.debug(f"Comparing parsing methods for {url}")
    # Order matters: try more reliable/focused methods first
    method_results_map = {
        "readability": method3_readability(url),
        "newspaper": method2_newspaper(url),
        "bs4": method1_bs4(url),
        "direct": method4_direct_extraction(url),
    }
    
    best_result = ""
    best_length = 0
    method_used = "None"

    # Prioritize readability and newspaper if they give substantial content
    for method_name in ["readability", "newspaper"]:
        result = method_results_map[method_name]
        if result and not result.startswith("Error") and len(result) > 200: # Prefer longer, non-error results
            tool_link_parser_logger.info(f"Selected method '{method_name}' for {url}, length: {len(result)}")
            return result, method_name

    # If not, check all results for the longest non-error one
    for method_name, result in method_results_map.items():
        if result and not result.startswith("Error"):
            if len(result) > best_length:
                best_length = len(result)
                best_result = result
                method_used = method_name
    
    if best_result:
        tool_link_parser_logger.info(f"Selected method '{method_used}' (longest non-error) for {url}, length: {best_length}")
        return best_result, method_used

    # If all methods resulted in errors or empty strings, return the first error encountered or an empty string
    tool_link_parser_logger.warning(f"All parsing methods failed or returned empty for {url}. Returning first error or empty.")
    return method_results_map["readability"], "readability (failed)" # Or any other default error string

def clean_text(text: str) -> str:
    """Additional text cleaning from unwanted elements."""
    if not text or text.startswith("Error"):
        return text # Don't clean error messages

    text = re.sub(r'\\n{2,}', '\\n\\n', text.strip()) # Consolidate multiple newlines after initial strip
    
    patterns_to_remove = [
        r"Subscribe to.*", r"Read also:.*", r"Share.*", 
        r"Comments.*", r"Copyright ©.*", r"\\d+ comments.*",
        r"Advertisement.*", r"Loading comments.*",
        r"Popular:.*", r"Related topics:.*", r"Source:.*",
        r"^\s*Menu\s*$", r"^\s*Search\s*$", r"Go to content",
        r"Cookie Policy.*", r"Privacy Policy.*", r"Terms of Service.*",
        r"Related Articles.*", r"Read More.*", r"Share this article.*", 
        r"Follow us on.*", r"Sign up for.*", r"Subscribe to.*",
        # Add more specific patterns if needed
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove lines that are too short or likely navigation/boilerplate
    lines = text.split('\\n')
    meaningful_lines = []
    
    # Special handling for price predictions - don't filter out important short lines with prices
    contains_price_data = "PRICE PREDICTION TABLES:" in text or "PRICE PREDICTION INFORMATION:" in text
    
    for line in lines:
        line_stripped = line.strip()
        # Always keep designated price prediction sections
        if "PRICE PREDICTION" in line_stripped:
            meaningful_lines.append(line)
            continue
            
        # For price prediction content, keep lines with price indicators
        if contains_price_data and any(indicator in line_stripped for indicator in ['$', '₿', '%', '|']):
            meaningful_lines.append(line)
            continue
            
        # Otherwise apply standard filtering
        if len(line_stripped) > 25 or '.' in line_stripped or ',' in line_stripped: 
            meaningful_lines.append(line)
    
    text = '\\n\\n'.join(meaningful_lines).strip()
    
    return text

@tool(
    name="parse_webpage",
    description="Parses the main textual content from a given URL. It tries several methods to extract the best possible text, with special handling for cryptocurrency price prediction pages.",
    parameters={
        "url": {
            "type": "string",
            "description": "The URL of the webpage to parse.",
            "required": True
        }
    }
)
def parse_webpage_content(url: str) -> str:
    """
    Main function to be registered as a tool.
    Parses the content from the given URL using a robust multi-method approach.
    Has specialized extraction for cryptocurrency price prediction pages.
    """
    tool_link_parser_logger.info(f"Tool 'parse_webpage' called with URL: {url}")
    if not url or not is_valid_url(url):
        error_msg = f"Invalid or empty URL provided to 'parse_webpage' tool: {url}"
        tool_link_parser_logger.error(error_msg)
        return f"Error: {error_msg}" # Return error string for ToolManager

    try:
        # Using compare_methods to get the best result and the method used
        content, method_used = compare_methods(url)
        original_length = len(content if content else '')
        tool_link_parser_logger.info(f"Raw content extracted from {url} using method '{method_used}'. Length: {original_length}")

        if not content or content.startswith("Error"):
            # If compare_methods returned an error or empty string
            error_msg = f"Failed to extract content from {url} using all methods. Last method tried: {method_used}. Result: {content}"
            tool_link_parser_logger.error(error_msg)
            return f"Error: {error_msg}" # Return error string

        # Clean the extracted text
        cleaned_content = clean_text(content)
        cleaned_length = len(cleaned_content)
        tool_link_parser_logger.info(f"Cleaned content from {url}. Original length: {original_length}, Cleaned length: {cleaned_length}")

        # If content is too minimal after cleaning (a common problem), use original
        if cleaned_length < 200 and original_length > 1000:
            tool_link_parser_logger.warning(f"Cleaned content too short ({cleaned_length} chars), using original content")
            # Try a lighter cleaning for original content
            lighter_cleaned = re.sub(r'\\n{3,}', '\\n\\n', content.strip())
            return lighter_cleaned

        if not cleaned_content:
            # If cleaning resulted in empty content, but original was not an error
            tool_link_parser_logger.warning(f"Content for {url} became empty after cleaning. Original was (method {method_used}): {content[:200]}...")
            return f"Warning: Content for {url} became empty after cleaning. Original content was not an error."
            
        return cleaned_content
    except Exception as e:
        # Catch-all for any unexpected error during the process for this URL
        error_msg = f"Unexpected critical error in 'parse_webpage' tool for URL {url}: {str(e)}"
        tool_link_parser_logger.exception(error_msg) # Log with stack trace
        return f"Error: {error_msg}" # Return error string