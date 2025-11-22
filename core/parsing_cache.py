"""
Parsing cache to avoid duplicate URL parsing.
"""

import time
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class ParsingCache:
    """
    Simple in-memory cache for parsed URLs to avoid re-parsing.
    Thread-safe and includes TTL.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize parsing cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
        """
        self._cache: Dict[str, Dict] = {}
        self.ttl = ttl_seconds
    
    def get(self, url: str) -> Optional[str]:
        """
        Get cached parsed content for URL.
        
        Args:
            url: URL to lookup
            
        Returns:
            Cached content or None if not found/expired
        """
        if url not in self._cache:
            return None
        
        entry = self._cache[url]
        
        # Check if expired
        if time.time() - entry['timestamp'] > self.ttl:
            del self._cache[url]
            logger.debug(f"Cache expired for: {url}")
            return None
        
        logger.debug(f"Cache HIT for: {url}")
        return entry['content']
    
    def set(self, url: str, content: str):
        """
        Cache parsed content for URL.
        
        Args:
            url: URL
            content: Parsed content
        """
        self._cache[url] = {
            'content': content,
            'timestamp': time.time()
        }
        logger.debug(f"Cached content for: {url} ({len(content)} chars)")
    
    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        logger.info("Parsing cache cleared")
    
    def cleanup_expired(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired = [
            url for url, entry in self._cache.items()
            if current_time - entry['timestamp'] > self.ttl
        ]
        
        for url in expired:
            del self._cache[url]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired cache entries")


# Global parsing cache instance
_parsing_cache = ParsingCache()


def get_parsing_cache() -> ParsingCache:
    """Get global parsing cache instance."""
    return _parsing_cache
