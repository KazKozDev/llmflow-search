import logging
from typing import Optional
from .base import CacheProvider
from .sqlite_cache import SQLiteCache

class CacheFactory:
    """Factory for creating cache providers."""
    
    @staticmethod
    def create(config: dict) -> CacheProvider:
        """
        Create a cache provider based on configuration.
        
        Args:
            config: Cache configuration dictionary
            
        Returns:
            CacheProvider instance
        """
        logger = logging.getLogger(__name__)
        
        provider = config.get('provider', 'sqlite').lower()
        
        if provider == 'sqlite':
            db_path = config.get('sqlite_path', './data/cache.db')
            ttl = config.get('ttl_seconds', 86400)
            compress = config.get('compress', True)
            
            logger.info(f"Creating SQLite cache at {db_path} with TTL={ttl}s")
            return SQLiteCache(db_path, ttl, compress)
        
        elif provider == 'redis':
            # Future: Redis implementation
            raise NotImplementedError("Redis cache not yet implemented")
        
        else:
            raise ValueError(f"Unknown cache provider: {provider}")
