import aiosqlite
import json
import zlib
import time
import logging
from typing import Any, Optional
from datetime import datetime
from .base import CacheProvider

class SQLiteCache(CacheProvider):
    """SQLite-based cache provider with TTL and compression."""
    
    def __init__(self, db_path: str, ttl_seconds: int = 86400, compress: bool = True):
        """
        Initialize SQLite cache.
        
        Args:
            db_path: Path to SQLite database file
            ttl_seconds: Default TTL in seconds
            compress: Whether to compress cached values
        """
        self.db_path = db_path
        self.default_ttl = ttl_seconds
        self.compress = compress
        self.logger = logging.getLogger(__name__)
        self.db = None
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    async def _init_db(self):
        """Initialize database connection and schema."""
        if self.db is None:
            self.db = await aiosqlite.connect(self.db_path)
            await self.db.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    expires_at REAL,
                    created_at REAL,
                    compressed INTEGER
                )
            ''')
            await self.db.execute('''
                CREATE INDEX IF NOT EXISTS idx_expires 
                ON cache(expires_at)
            ''')
            await self.db.commit()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        await self._init_db()
        
        try:
            cursor = await self.db.execute(
                'SELECT value, expires_at, compressed FROM cache WHERE key = ?',
                (key,)
            )
            row = await cursor.fetchone()
            
            if row is None:
                self.misses += 1
                return None
            
            value_blob, expires_at, compressed = row
            
            # Check expiration
            if expires_at and time.time() > expires_at:
                await self.delete(key)
                self.misses += 1
                return None
            
            # Decompress if needed
            if compressed:
                value_blob = zlib.decompress(value_blob)
            
            # Deserialize
            value = json.loads(value_blob.decode('utf-8'))
            self.hits += 1
            return value
            
        except Exception as e:
            self.logger.error(f"Error getting cache key {key}: {e}")
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> bool:
        """Set value in cache."""
        await self._init_db()
        
        try:
            # Serialize
            value_json = json.dumps(value).encode('utf-8')
            
            # Compress if enabled
            compressed = 0
            if self.compress and len(value_json) > 1024:  # Only compress if > 1KB
                value_json = zlib.compress(value_json)
                compressed = 1
            
            # Calculate expiration
            ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
            expires_at = time.time() + ttl if ttl > 0 else None
            
            await self.db.execute('''
                INSERT OR REPLACE INTO cache (key, value, expires_at, created_at, compressed)
                VALUES (?, ?, ?, ?, ?)
            ''', (key, value_json, expires_at, time.time(), compressed))
            
            await self.db.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        await self._init_db()
        
        try:
            await self.db.execute('DELETE FROM cache WHERE key = ?', (key,))
            await self.db.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cached values."""
        await self._init_db()
        
        try:
            await self.db.execute('DELETE FROM cache')
            await self.db.commit()
            self.hits = 0
            self.misses = 0
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
    
    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.
        
        Returns:
            Number of entries removed
        """
        await self._init_db()
        
        try:
            cursor = await self.db.execute(
                'DELETE FROM cache WHERE expires_at IS NOT NULL AND expires_at < ?',
                (time.time(),)
            )
            await self.db.commit()
            return cursor.rowcount
        except Exception as e:
            self.logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    async def get_stats(self) -> dict:
        """Get cache statistics."""
        await self._init_db()
        
        try:
            # Get count and size
            cursor = await self.db.execute('''
                SELECT 
                    COUNT(*) as count,
                    SUM(LENGTH(value)) as size_bytes
                FROM cache
            ''')
            row = await cursor.fetchone()
            count, size_bytes = row if row else (0, 0)
            
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'total_entries': count,
                'size_bytes': size_bytes or 0,
                'size_mb': (size_bytes or 0) / 1024 / 1024
            }
        except Exception as e:
            self.logger.error(f"Error getting cache stats: {e}")
            return {}
    
    async def close(self):
        """Close database connection."""
        if self.db:
            await self.db.close()
            self.db = None
