import time
import asyncio
import logging
from typing import Dict, Optional
from collections import defaultdict

class RateLimiter:
    """
    Rate limiter using token bucket algorithm.
    Supports per-tool rate limits with automatic refill.
    """
    
    def __init__(self, config: Dict[str, dict]):
        """
        Initialize rate limiter.
        
        Args:
            config: Dictionary mapping tool names to rate limit configs
                   Format: {"tool_name": {"requests_per_minute": 60}}
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Token buckets: {tool_name: {"tokens": float, "last_refill": float, "capacity": int}}
        self.buckets: Dict[str, dict] = defaultdict(lambda: {
            "tokens": 0,
            "last_refill": time.time(),
            "capacity": 0
        })
        
        # Initialize buckets
        for tool_name, limits in config.items():
            rpm = limits.get("requests_per_minute", 30)
            self.buckets[tool_name] = {
                "tokens": rpm,
                "last_refill": time.time(),
                "capacity": rpm
            }
        
        self.logger.info(f"Rate limiter initialized with {len(config)} tool limits")
    
    def _refill_tokens(self, tool_name: str):
        """Refill tokens based on elapsed time."""
        bucket = self.buckets[tool_name]
        now = time.time()
        elapsed = now - bucket["last_refill"]
        
        # Add tokens based on elapsed time (tokens per second = capacity / 60)
        tokens_to_add = (bucket["capacity"] / 60.0) * elapsed
        bucket["tokens"] = min(bucket["capacity"], bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
    
    async def acquire(self, tool_name: str, wait: bool = True) -> bool:
        """
        Acquire a token for the given tool.
        
        Args:
            tool_name: Name of the tool
            wait: If True, wait until a token is available
            
        Returns:
            True if token acquired, False if wait=False and no token available
        """
        # Get config for this tool or use default
        if tool_name not in self.config:
            if "default" in self.config:
                tool_config = self.config["default"]
                if tool_name not in self.buckets:
                    rpm = tool_config.get("requests_per_minute", 30)
                    self.buckets[tool_name] = {
                        "tokens": rpm,
                        "last_refill": time.time(),
                        "capacity": rpm
                    }
            else:
                # No limit configured
                return True
        
        bucket = self.buckets[tool_name]
        
        while True:
            self._refill_tokens(tool_name)
            
            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True
            
            if not wait:
                self.logger.warning(f"Rate limit exceeded for {tool_name}")
                return False
            
            # Wait for next token (calculate wait time)
            wait_time = (1.0 - bucket["tokens"]) / (bucket["capacity"] / 60.0)
            self.logger.debug(f"Rate limit hit for {tool_name}, waiting {wait_time:.2f}s")
            await asyncio.sleep(wait_time)
    
    def get_stats(self) -> dict:
        """Get rate limiter statistics."""
        stats = {}
        for tool_name, bucket in self.buckets.items():
            self._refill_tokens(tool_name)
            stats[tool_name] = {
                "available_tokens": int(bucket["tokens"]),
                "capacity": bucket["capacity"],
                "utilization": 1.0 - (bucket["tokens"] / bucket["capacity"])
            }
        return stats
    
    def reset(self, tool_name: Optional[str] = None):
        """
        Reset rate limiter.
        
        Args:
            tool_name: If specified, reset only this tool. Otherwise reset all.
        """
        if tool_name:
            if tool_name in self.buckets:
                self.buckets[tool_name]["tokens"] = self.buckets[tool_name]["capacity"]
                self.buckets[tool_name]["last_refill"] = time.time()
        else:
            for bucket in self.buckets.values():
                bucket["tokens"] = bucket["capacity"]
                bucket["last_refill"] = time.time()
