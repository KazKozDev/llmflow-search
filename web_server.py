#!/usr/bin/env python3
"""
LLMFlow Search Agent - Web Server
FastAPI server with WebSocket for real-time search updates.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Import agent components
from core.config import AppConfig, load_config
from core.agent_factory import AgentFactory
from core.event_bus import Event
from core.monitoring import get_metrics
from core.background_jobs import get_job_queue, JobStatus, Job
from core.memory_module import MemoryModule
from core.llm_service import LLMService
from core.planning_module import PlanningModule
from core.tools_module import ToolsModule
from core.report_generator import ReportGenerator
from core.agent_core import AgentCore
from core.search_intent_analyzer import SearchIntentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active sessions
active_sessions: Dict[str, dict] = {}

# Configuration (validated)
config = load_config('config.json')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Initializing Agent Factory...")
    await AgentFactory.initialize(config)
    logger.info("Agent Factory initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Agent Factory...")
    await AgentFactory.shutdown()
    logger.info("Shutdown complete")


app = FastAPI(title="LLMFlow Search Agent API", lifespan=lifespan)

# Mount static files
app.mount("/static", StaticFiles(directory="web/static"), name="static")

# Store active sessions
active_sessions: Dict[str, dict] = {}


class SearchRequest(BaseModel):
    query: str
    max_iterations: int = 10
    mode: str = "standard"  # "standard" or "deep"


class SearchResponse(BaseModel):
    session_id: str
    status: str
    job_id: str = None
    message: str = None


@app.get("/")
async def root():
    """Serve the main UI."""
    return FileResponse("web/static/index.html")


@app.post("/api/search", response_model=SearchResponse)
async def start_search(request: SearchRequest):
    """Start a new search session or background job."""
    
    # Handle Deep Search (Background Job)
    if request.mode == "deep":
        job_queue = get_job_queue()
        
        # Import worker here to avoid circular imports
        from core.deep_search_worker import run_deep_search_worker
        
        # Define the task wrapper (receives Job object as parameter)
        async def worker(job: Job):
            await run_deep_search_worker(
                job=job,
                config=config,
                query=request.query,
                max_iterations=max(request.max_iterations, 30)
            )
            return job.result  # Return result for JobQueue
            
        # Submit job - note: worker receives job, query is metadata
        job_id = await job_queue.submit(
            query=request.query,
            worker=worker
        )
        
        return SearchResponse(
            session_id="",
            status="queued",
            job_id=job_id,
            message="Deep search job started in background"
        )

    # Standard Search (WebSocket)
    session_id = str(uuid.uuid4())
    active_sessions[session_id] = {
        "query": request.query,
        "max_iterations": request.max_iterations,
        "created_at": datetime.now().isoformat(),
        "status": "initialized"
    }
    
    logger.info(f"Created new standard session: {session_id}")
    return SearchResponse(session_id=session_id, status="initialized")


@app.websocket("/ws/search/{session_id}")
async def websocket_search(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for streaming search results."""
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close()
        return
    
    session = active_sessions[session_id]
    query = session["query"]
    max_iterations = session["max_iterations"]
    
    try:
        # Send initial status
        await websocket.send_json({"type": "status", "message": "Initializing agent..."})
        
        # Create agent using factory (FAST - reuses shared resources!)
        agent = await AgentFactory.create_agent(
            max_iterations=max_iterations,
            enable_intent_analyzer=config.intent_analyzer.enabled,
            enable_events=True
        )
        
        # Subscribe to agent events for real-time updates
        async def send_progress(event: Event):
            await websocket.send_json({
                "type": "progress",
                "event": event.type,
                "data": event.data,
                "timestamp": event.timestamp
            })
        
        # Register event handlers
        if hasattr(agent, 'events'):
            agent.events.on("step.started", send_progress)
            agent.events.on("step.completed", send_progress)
            agent.events.on("tool.executed", send_progress)
        
        await websocket.send_json({"type": "status", "message": f"Processing query: {query}"})
        
        # Process query
        report = await agent.process_query(query)
        
        # Send results
        await websocket.send_json({
            "type": "result",
            "report": report,
            "sources": list(agent.memory.get_links().items())
        })
        
        await websocket.send_json({"type": "complete", "message": "Search completed"})
        session["status"] = "completed"
        
    except WebSocketDisconnect:
        logger.info(f"Client disconnected from session {session_id}")
    except Exception as e:
        logger.error(f"Error in search session {session_id}: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        await websocket.close()


@app.get("/api/sessions")
async def list_sessions():
    """List all search sessions."""
    return {"sessions": active_sessions}


@app.get("/api/tools")
async def list_tools():
    """List available search tools."""
    return {
        "tools": [
            {"name": "search_duckduckgo", "description": "Search DuckDuckGo"},
            {"name": "search_wikipedia", "description": "Search Wikipedia"},
            {"name": "search_searxng", "description": "Meta-search via SearXNG"},
            {"name": "search_arxiv", "description": "Search scientific papers"},
            {"name": "search_youtube", "description": "Search YouTube videos"},
            {"name": "search_pubmed", "description": "Search biomedical literature"},
            {"name": "search_gutenberg", "description": "Search free books"},
            {"name": "search_openstreetmap", "description": "Geocoding search"},
            {"name": "search_wayback", "description": "Internet Archive"},
        ]
    }


@app.get("/api/metrics")
async def get_system_metrics():
    """Get system metrics."""
    metrics = get_metrics()
    factory_metrics = AgentFactory.get_metrics()
    
    return {
        "system": metrics.get_stats(),
        "llm": factory_metrics,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/jobs")
async def list_background_jobs(status: str = None):
    """List background jobs."""
    queue = get_job_queue()
    
    job_status = JobStatus(status) if status else None
    jobs = queue.list_jobs(status=job_status)
    
    return {
        "jobs": [job.to_dict() for job in jobs],
        "stats": queue.get_stats()
    }


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get specific job status."""
    queue = get_job_queue()
    job = queue.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_background_job(job_id: str):
    """Cancel a running job."""
    queue = get_job_queue()
    success = await queue.cancel_job(job_id)
    
    if not success:
        raise HTTPException(status_code=400, detail="Job not running or not found")
    
    return {"status": "cancelled", "job_id": job_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
