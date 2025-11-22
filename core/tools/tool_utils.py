"""
Shared utilities for search tools.
Includes retry logic, timeout handling, and error formatting.
"""

import asyncio
import aiohttp
import logging
from typing import Any, Callable, Dict, List, Optional
from functools import wraps

logger = logging.getLogger(__name__)

# Default timeout for all tool API calls
DEFAULT_TIMEOUT = 7

class ToolError(Exception):
    """Custom exception for tool errors."""
    pass


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 10.0
):
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        
    Returns:
        Result from successful function call
        
    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return await func()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exception = e
            
            if attempt < max_retries - 1:
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries} after {delay}s. Error: {e}"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(f"All {max_retries} retries failed: {e}")
    
    # All retries failed
    raise last_exception


def format_error(error: Exception, tool_name: str, query: str = "") -> List[Dict[str, Any]]:
    """
    Format error in standard structure.
    
    Args:
        error: Exception that occurred
        tool_name: Name of the tool
        query: Optional query that caused the error
        
    Returns:
        List with single error dict
    """
    error_msg = str(error)
    
    if isinstance(error, asyncio.TimeoutError):
        error_msg = "Request timed out"
    elif isinstance(error, aiohttp.ClientError):
        error_msg = f"Network error: {error_msg}"
    
    logger.error(f"{tool_name} error for query '{query}': {error_msg}")
    
    return [{
        "error": error_msg,
        "tool": tool_name,
        "query": query,
        "type": type(error).__name__
    }]


async def fetch_with_timeout(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict] = None,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs
) -> Dict[str, Any]:
    """
    Fetch URL with timeout and error handling.
    
    Args:
        session: aiohttp session
        url: URL to fetch
        params: Query parameters
        timeout: Timeout in seconds
        **kwargs: Additional arguments for session.get
        
    Returns:
        Parsed JSON response
        
    Raises:
        ToolError: If request fails or returns non-200
    """
    try:
        async with session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout),
            **kwargs
        ) as response:
            if response.status != 200:
                raise ToolError(f"HTTP {response.status}")
            
            return await response.json()
    except asyncio.TimeoutError:
        raise ToolError(f"Timeout after {timeout}s")
    except aiohttp.ClientError as e:
        raise ToolError(f"Network error: {e}")


def measure_time(func):
    """Decorator to measure and log execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        import time
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper


def safe_get(data: Dict, *keys, default=None):
    """
    Safely get nested dictionary values.
    
    Example:
        safe_get(data, 'a', 'b', 'c', default='')
        # Returns data['a']['b']['c'] or '' if any key missing
    """
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, {})
        elif isinstance(data, list) and isinstance(key, int) and key < len(data):
            data = data[key]
        else:
            return default
    return data if data != {} else default
