"""
LLM Gateway - Centralized entry point for all LLM calls.
Provides caching, metrics tracking, and fallback support.
"""

import asyncio
import hashlib
import logging
from typing import Optional, Dict, Any
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class LLMMetrics:
    """Track LLM usage metrics."""
    
    def __init__(self):
        self.total_calls = 0
        self.total_tokens = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.errors = 0
        self.call_times = []
    
    def record_call(self, tokens: int = 0, duration: float = 0, from_cache: bool = False):
        """Record an LLM call."""
        self.total_calls += 1
        self.total_tokens += tokens
        self.call_times.append(duration)
        
        if from_cache:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
    
    def record_error(self):
        """Record an error."""
        self.errors += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics."""
        avg_time = sum(self.call_times) / len(self.call_times) if self.call_times else 0
        cache_rate = self.cache_hits / self.total_calls if self.total_calls > 0 else 0
        
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": f"{cache_rate:.1%}",
            "errors": self.errors,
            "avg_call_time": f"{avg_time:.2f}s"
        }


class LLMGateway:
    """
    Centralized gateway for all LLM interactions.
    
    Features:
    - Response caching
    - Metrics tracking
    - Fallback LLM support
    - Automatic retries
    """
    
    def __init__(
        self,
        primary_llm,
        cache=None,
        fallback_llm=None,
        enable_metrics: bool = True
    ):
        """
        Initialize LLM Gateway.
        
        Args:
            primary_llm: Primary LLM service
            cache: Optional cache for responses
            fallback_llm: Optional fallback LLM service
            enable_metrics: Enable metrics tracking
        """
        self.primary = primary_llm
        self.fallback = fallback_llm
        self.cache = cache
        self.metrics = LLMMetrics() if enable_metrics else None
        
        logger.info("LLM Gateway initialized")
    
    def _make_cache_key(self, prompt: str, system: str, context: str) -> str:
        """Generate cache key from inputs."""
        combined = f"{context}:{system}:{prompt}"
        return f"llm:{hashlib.md5(combined.encode()).hexdigest()}"
    
    async def generate(
        self,
        prompt: str,
        system: str = "",
        context: str = "default",
        use_cache: bool = True,
        use_fallback: bool = True
    ) -> str:
        """
        Generate LLM response with caching and fallback.
        
        Args:
            prompt: User prompt
            system: System message
            context: Context for caching (e.g., "planning", "parsing")
            use_cache: Enable cache lookup
            use_fallback: Use fallback LLM on error
            
        Returns:
            LLM response
        """
        start_time = time.time()
        cache_key = self._make_cache_key(prompt, system, context)
        
        # Try cache first
        if use_cache and self.cache:
            try:
                cached = await self.cache.get(cache_key)
                if cached:
                    duration = time.time() - start_time
                    if self.metrics:
                        self.metrics.record_call(duration=duration, from_cache=True)
                    logger.debug(f"Cache hit for context '{context}'")
                    return cached
            except Exception as e:
                logger.warning(f"Cache read error: {e}")
        
        # Call primary LLM
        try:
            response = await self._call_llm(self.primary, prompt, system)
            duration = time.time() - start_time
            
            # Cache response
            if use_cache and self.cache:
                try:
                    await self.cache.set(cache_key, response)
                except Exception as e:
                    logger.warning(f"Cache write error: {e}")
            
            if self.metrics:
                self.metrics.record_call(
                    tokens=len(response.split()),  # Rough estimate
                    duration=duration,
                    from_cache=False
                )
            
            return response
            
        except Exception as e:
            logger.error(f"Primary LLM failed: {e}")
            
            if self.metrics:
                self.metrics.record_error()
            
            # Try fallback LLM
            if use_fallback and self.fallback:
                logger.info("Attempting fallback LLM")
                try:
                    response = await self._call_llm(self.fallback, prompt, system)
                    duration = time.time() - start_time
                    
                    if self.metrics:
                        self.metrics.record_call(duration=duration, from_cache=False)
                    
                    return response
                except Exception as fallback_error:
                    logger.error(f"Fallback LLM also failed: {fallback_error}")
            
            # Ultimate fallback: simple response
            return self._simple_fallback(prompt)
    
    async def _call_llm(self, llm_service, prompt: str, system: str) -> str:
        """Call LLM service (async)."""
        if hasattr(llm_service, 'generate_response_async'):
            return await llm_service.generate_response_async(prompt, system)
        else:
            # Fallback to sync method
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                llm_service.generate_response,
                prompt,
                system
            )
    
    def _simple_fallback(self, prompt: str) -> str:
        """Provide simple fallback response when all LLMs fail."""
        logger.warning("Using simple fallback response")
        return "I apologize, but I'm unable to process this request at the moment. Please try again later."
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        if self.metrics:
            return self.metrics.get_stats()
        return {}
