"""
Agent Factory - Manages agent lifecycle and shared resources.
Provides singleton pattern for expensive resources and agent pooling.
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from core.config import AppConfig
from core.llm_service import LLMService
from core.llm_gateway import LLMGateway
from core.memory_module import MemoryModule
from core.planning_module import PlanningModule
from core.search_intent_analyzer import SearchIntentAnalyzer
from core.tools_module import ToolsModule
from core.report_generator import ReportGenerator
from core.agent_core import AgentCore
from core.event_bus import EventBus
from core.caching.factory import CacheFactory

logger = logging.getLogger(__name__)


class AgentFactory:
    """
    Factory for creating and managing agent instances.
    
    Uses singleton pattern for shared expensive resources:
    - Embedding models
    - Cache instances
    - LLM services (optional)
    """
    
    _shared_resources: Optional[Dict[str, Any]] = None
    _initialization_lock = asyncio.Lock()
    
    @classmethod
    async def initialize(cls, config: AppConfig):
        """
        Initialize shared resources (call once at startup).
        
        Args:
            config: Application configuration
        """
        async with cls._initialization_lock:
            if cls._shared_resources is not None:
                logger.warning("Shared resources already initialized")
                return
            
            logger.info("Initializing shared resources...")
            
            # Initialize cache (expensive I/O)
            cache = CacheFactory.create(config.cache.model_dump())
            
            # Initialize LLM service
            llm_service = LLMService(
                provider=config.llm.provider,
                model=config.llm.model,
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            )
            
            # Initialize LLM Gateway with cache
            llm_gateway = LLMGateway(
                primary_llm=llm_service,
                cache=cache,
                fallback_llm=None,  # Can add fallback LLM here
                enable_metrics=True
            )
            
            # Store shared resources
            cls._shared_resources = {
                "config": config,
                "cache": cache,
                "llm_service": llm_service,
                "llm_gateway": llm_gateway
            }
            
            logger.info("Shared resources initialized successfully")
    
    @classmethod
    async def create_agent(
        cls,
        max_iterations: int = 10,
        enable_intent_analyzer: bool = True,
        enable_events: bool = True
    ) -> AgentCore:
        """
        Create a new agent instance with shared resources.
        
        Args:
            max_iterations: Maximum search iterations
            enable_intent_analyzer: Enable search intent analysis
            enable_events: Enable event bus
            
        Returns:
            Configured AgentCore instance
        """
        if cls._shared_resources is None:
            raise RuntimeError("AgentFactory not initialized. Call initialize() first.")
        
        config = cls._shared_resources["config"]
        cache = cls._shared_resources["cache"]
        llm_gateway = cls._shared_resources["llm_gateway"]
        llm_service = cls._shared_resources["llm_service"]
        
        # Create per-agent components (lightweight)
        
        # Memory module (shares embedding model via singleton)
        memory = MemoryModule(memory_path=config.memory.path)
        
        # Search intent analyzer (optional)
        intent_analyzer = None
        if enable_intent_analyzer and config.intent_analyzer.enabled:
            intent_analyzer = SearchIntentAnalyzer(llm_gateway)  # Use gateway
        
        # Planning module
        planning = PlanningModule(llm_gateway, search_intent_analyzer=intent_analyzer)  # Use gateway
        
        # Tools module (shares cache)
        tools = ToolsModule(
            memory=memory,
            llm_service=llm_gateway,  # Use gateway
            config=config.model_dump(),
            max_results=config.search.max_results,
            safe_search=True,
            parse_top_results=config.search.parse_top_results
        )
        
        # Report generator
        report_generator = ReportGenerator(memory, llm_gateway)  # Use gateway

        
        # Event bus (optional)
        event_bus = EventBus(keep_history=True) if enable_events else None
        
        # Create agent
        agent = AgentCore(
            memory=memory,
            planning=planning,
            tools=tools,
            report_generator=report_generator,
            llm_service=llm_service,
            max_iterations=max_iterations
        )
        
        # Add event bus to agent if enabled
        if event_bus:
            agent.events = event_bus
        
        logger.debug(f"Created new agent instance (iterations={max_iterations})")
        
        return agent
    
    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """Get metrics from shared resources."""
        if cls._shared_resources is None:
            return {}
        
        llm_gateway = cls._shared_resources.get("llm_gateway")
        if llm_gateway:
            return llm_gateway.get_metrics()
        
        return {}
    
    @classmethod
    async def shutdown(cls):
        """Cleanup shared resources."""
        if cls._shared_resources is None:
            return
        
        logger.info("Shutting down shared resources...")
        
        cache = cls._shared_resources.get("cache")
        if cache and hasattr(cache, 'close'):
            await cache.close()
        
        cls._shared_resources = None
        logger.info("Shared resources cleaned up")
