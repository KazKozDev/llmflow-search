"""
Deep Search Worker - Handles autonomous background search tasks.
"""
import asyncio
import logging
import os
import time
from typing import Dict, Any, Optional
from datetime import datetime

from core.config import AppConfig
from core.agent_factory import AgentFactory
from core.background_jobs import Job, JobStatus

logger = logging.getLogger(__name__)

async def run_deep_search_worker(job: Job, config: AppConfig, query: str, max_iterations: int = 30):
    """
    Worker function for Deep Search - runs agent autonomously with extended iterations.
    
    Deep Search differences from standard search:
    - Higher iteration limit (30+ vs 10)
    - Results saved to file
    - Can run in background
    - Agent uses LLM to understand query and select appropriate tools
    
    Args:
        job: The background job object
        config: Application configuration
        query: The search query (agent will understand it via LLM)
        max_iterations: Maximum iterations for deep search (default: 30)
    """
    logger.info(f"Starting deep search worker for job {job.id} with query: {query}")
    
    try:
        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.progress = 10
        job.message = "Initializing agent..."
        
        # Create agent with extended capabilities
        agent = await AgentFactory.create_agent(
            max_iterations=max_iterations,  # More iterations for thorough research
            enable_intent_analyzer=True,    # Use LLM-based intent analysis
            enable_events=False
        )
        
        job.progress = 20
        job.message = "Agent initialized. Analyzing query and creating plan..."
        
        # Let the agent process the query
        # The Planning Module (via LLM) will:
        # 1. Understand the query ("100 ArXiv papers on X")
        # 2. Choose appropriate tools (ArXiv, PubMed, etc.)
        # 3. Execute searches
        # 4. Generate comprehensive report
        
        report = await agent.process_query(query)
        
        job.progress = 90
        job.message = "Search complete. Saving report..."
        
        # Save report to file
        reports_dir = "reports"
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join([c if c.isalnum() else "_" for c in query[:30]])
        filename = f"{reports_dir}/job_{job.id}_{safe_query}_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        
        # Store result metadata
        job.result = {
            "report_path": filename,
            "summary": report[:200] + "...",
        }
        
        job.status = JobStatus.COMPLETED
        job.completed_at = time.time()
        job.progress = 100
        job.message = f"Completed. Report saved to {filename}"
        
        logger.info(f"Deep search job {job.id} completed successfully")
        
    except Exception as e:
        logger.error(f"Deep search job {job.id} failed: {str(e)}", exc_info=True)
        job.status = JobStatus.FAILED
        job.completed_at = time.time()
        job.error = str(e)
        job.message = f"Failed: {str(e)}"
