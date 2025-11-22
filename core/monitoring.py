"""
Monitoring and Metrics - Observability for the application.
Tracks performance, usage, and health metrics.
"""

import time
import logging
from typing import Dict, Any, List
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: str
    tags: Dict[str, str]


class Metrics:
    """
    Centralized metrics tracking system.
    
    Features:
    - Counters (increment only)
    - Gauges (current value)
    - Timers (duration tracking)
    - Histograms (distribution)
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics system.
        
        Args:
            max_history: Maximum metric points to keep
        """
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.history: deque = deque(maxlen=max_history)
        self._locks = defaultdict(asyncio.Lock)
    
    def increment(self, metric: str, value: int = 1, tags: Dict[str, str] = None):
        """
        Increment a counter.
        
        Args:
            metric: Metric name
            value: Amount to increment
            tags: Optional tags
        """
        self.counters[metric] += value
        self._record(metric, self.counters[metric], tags or {})
    
    def gauge(self, metric: str, value: float, tags: Dict[str, str] = None):
        """
        Set a gauge value.
        
        Args:
            metric: Metric name
            value: Current value
            tags: Optional tags
        """
        self.gauges[metric] = value
        self._record(metric, value, tags or {})
    
    def timing(self, metric: str, duration: float, tags: Dict[str, str] = None):
        """
        Record a timing.
        
        Args:
            metric: Metric name
            duration: Duration in seconds
            tags: Optional tags
        """
        self.timers[metric].append(duration)
        # Keep only last 100 timings per metric
        if len(self.timers[metric]) > 100:
            self.timers[metric] = self.timers[metric][-100:]
        
        self._record(metric, duration, tags or {})
    
    def _record(self, name: str, value: float, tags: Dict[str, str]):
        """Record metric point in history."""
        point = MetricPoint(
            name=name,
            value=value,
            timestamp=datetime.now().isoformat(),
            tags=tags
        )
        self.history.append(point)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get all current statistics."""
        stats = {
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timers": {}
        }
        
        # Calculate timer statistics
        for metric, timings in self.timers.items():
            if timings:
                stats["timers"][metric] = {
                    "count": len(timings),
                    "avg": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "p95": self._percentile(timings, 0.95),
                    "p99": self._percentile(timings, 0.99)
                }
        
        return stats
    
    def get_metric(self, metric: str) -> Any:
        """Get specific metric value."""
        if metric in self.counters:
            return self.counters[metric]
        if metric in self.gauges:
            return self.gauges[metric]
        if metric in self.timers:
            timings = self.timers[metric]
            return sum(timings) / len(timings) if timings else 0
        return None
    
    def get_history(self, metric: str = None, limit: int = 100) -> List[Dict]:
        """
        Get metric history.
        
        Args:
            metric: Filter by metric name (optional)
            limit: Maximum points to return
            
        Returns:
            List of metric points
        """
        points = self.history
        
        if metric:
            points = [p for p in points if p.name == metric]
        
        return [asdict(p) for p in list(points)[-limit:]]
    
    def reset(self):
        """Reset all metrics."""
        self.counters.clear()
        self.gauges.clear()
        self.timers.clear()
        self.history.clear()
        logger.info("All metrics reset")
    
    @staticmethod
    def _percentile(values: List[float], p: float) -> float:
        """Calculate percentile."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, metrics: Metrics, metric_name: str, tags: Dict[str, str] = None):
        self.metrics = metrics
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        self.metrics.timing(self.metric_name, duration, self.tags)


# Global metrics instance
_global_metrics = Metrics()


def get_metrics() -> Metrics:
    """Get global metrics instance."""
    return _global_metrics


def track_time(metric_name: str, tags: Dict[str, str] = None):
    """
    Decorator to track function execution time.
    
    Usage:
        @track_time("my_function")
        def my_function():
            ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            with Timer(get_metrics(), metric_name, tags):
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with Timer(get_metrics(), metric_name, tags):
                return func(*args, **kwargs)
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
