"""
Event Bus - Pub/Sub system for real-time updates.
Enables communication between components without tight coupling.
"""

import asyncio
import logging
from typing import Callable, Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data structure."""
    type: str
    data: Any
    timestamp: str
    source: str = "unknown"


class EventBus:
    """
    Simple async event bus for pub/sub communication.
    
    Features:
    - Async event handlers
    - Multiple listeners per event
    - Event history (optional)
    - Error handling
    """
    
    def __init__(self, keep_history: bool = False, max_history: int = 100):
        """
        Initialize Event Bus.
        
        Args:
            keep_history: Store event history
            max_history: Maximum events to keep in history
        """
        self.listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.keep_history = keep_history
        self.max_history = max_history
        self.history: List[Event] = []
        self._lock = asyncio.Lock()
    
    def on(self, event_type: str, handler: Callable):
        """
        Register event handler.
        
        Args:
            event_type: Event type to listen for
            handler: Async function to call when event occurs
        """
        self.listeners[event_type].append(handler)
        logger.debug(f"Registered handler for event '{event_type}'")
    
    def off(self, event_type: str, handler: Callable):
        """
        Unregister event handler.
        
        Args:
            event_type: Event type
            handler: Handler to remove
        """
        if handler in self.listeners[event_type]:
            self.listeners[event_type].remove(handler)
            logger.debug(f"Unregistered handler for event '{event_type}'")
    
    async def emit(self, event_type: str, data: Any = None, source: str = "unknown"):
        """
        Emit event to all listeners.
        
        Args:
            event_type: Type of event
            data: Event data
            source: Source component name
        """
        event = Event(
            type=event_type,
            data=data,
            timestamp=datetime.now().isoformat(),
            source=source
        )
        
        # Store in history
        if self.keep_history:
            async with self._lock:
                self.history.append(event)
                if len(self.history) > self.max_history:
                    self.history.pop(0)
        
        # Notify listeners
        handlers = self.listeners.get(event_type, [])
        if not handlers:
            logger.debug(f"No handlers for event '{event_type}'")
            return
        
        # Call all handlers concurrently
        tasks = []
        for handler in handlers:
            tasks.append(self._safe_call_handler(handler, event))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_call_handler(self, handler: Callable, event: Event):
        """Call handler with error handling."""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                # Sync handler - run in executor
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, handler, event)
        except Exception as e:
            logger.error(f"Error in event handler for '{event.type}': {e}")
    
    def get_history(self, event_type: str = None) -> List[Event]:
        """
        Get event history.
        
        Args:
            event_type: Filter by event type (optional)
            
        Returns:
            List of events
        """
        if not self.keep_history:
            return []
        
        if event_type:
            return [e for e in self.history if e.type == event_type]
        return self.history.copy()
    
    def clear_history(self):
        """Clear event history."""
        self.history.clear()
    
    def get_listener_count(self, event_type: str = None) -> int:
        """Get number of listeners."""
        if event_type:
            return len(self.listeners.get(event_type, []))
        return sum(len(handlers) for handlers in self.listeners.values())
