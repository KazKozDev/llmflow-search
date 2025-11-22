from abc import ABC, abstractmethod
from typing import Any, Optional
from datetime import datetime, timedelta

class CacheProvider(ABC):
    """Abstract base class for cache providers."""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all cached values.
        
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    async def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats (hits, misses, size, etc.)
        """
        pass
    
    @abstractmethod
    async def close(self):
        """Close cache connection."""
        pass
