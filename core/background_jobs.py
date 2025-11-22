"""
Background Jobs - Task queue for long-running searches.
Allows search to continue even if WebSocket disconnects.
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, Optional, Callable, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Background job representation."""
    id: str
    query: str
    status: JobStatus = JobStatus.PENDING
    progress: int = 0  # 0-100
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "query": self.query,
            "status": self.status,
            "progress": self.progress,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata
        }


class JobQueue:
    """
    Background job queue for long-running searches.
    
    Features:
    - Async job execution
    - Progress tracking
    - Job persistence (optional)
    - Automatic cleanup
    """
    
    def __init__(self, max_concurrent: int = 3, retention_hours: int = 24):
        """
        Initialize job queue.
        
        Args:
            max_concurrent: Maximum concurrent jobs
            retention_hours: Hours to keep completed jobs
        """
        self.jobs: Dict[str, Job] = {}
        self.max_concurrent = max_concurrent
        self.retention_hours = retention_hours
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"JobQueue initialized (max_concurrent={max_concurrent})")
    
    async def submit(
        self,
        query: str,
        worker: Callable,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Submit a job to the queue.
        
        Args:
            query: Search query
            worker: Async function to execute
            metadata: Optional metadata
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job = Job(
            id=job_id,
            query=query,
            metadata=metadata or {}
        )
        
        self.jobs[job_id] = job
        
        # Start job execution
        task = asyncio.create_task(self._execute_job(job_id, worker))
        self.running_tasks[job_id] = task
        
        logger.info(f"Job submitted: {job_id} (query: {query})")
        
        return job_id
    
    async def _execute_job(self, job_id: str, worker: Callable):
        """Execute job with concurrency control."""
        job = self.jobs[job_id]
        
        async with self._semaphore:
            try:
                job.status = JobStatus.RUNNING
                job.started_at = datetime.now().isoformat()
                
                logger.info(f"Job started: {job_id}")
                
                # Execute worker function
                result = await worker(job)
                
                job.status = JobStatus.COMPLETED
                job.result = result
                job.progress = 100
                job.completed_at = datetime.now().isoformat()
                
                logger.info(f"Job completed: {job_id}")
                
            except asyncio.CancelledError:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now().isoformat()
                logger.warning(f"Job cancelled: {job_id}")
                
            except Exception as e:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now().isoformat()
                logger.error(f"Job failed: {job_id} - {e}")
                
            finally:
                # Clean up task reference
                if job_id in self.running_tasks:
                    del self.running_tasks[job_id]
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[Job]:
        """
        List jobs with optional filtering.
        
        Args:
            status: Filter by status
            limit: Maximum jobs to return
            
        Returns:
            List of jobs
        """
        jobs = list(self.jobs.values())
        
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        return jobs[:limit]
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job.
        
        Args:
            job_id: Job ID
            
        Returns:
            True if cancelled, False if not found or not running
        """
        job = self.jobs.get(job_id)
        if not job or job.status != JobStatus.RUNNING:
            return False
        
        task = self.running_tasks.get(job_id)
        if task:
            task.cancel()
            logger.info(f"Job cancelled: {job_id}")
            return True
        
        return False
    
    def update_progress(self, job_id: str, progress: int):
        """
        Update job progress.
        
        Args:
            job_id: Job ID
            progress: Progress percentage (0-100)
        """
        job = self.jobs.get(job_id)
        if job:
            job.progress = max(0, min(100, progress))
    
    async def cleanup_old_jobs(self):
        """Remove old completed/failed jobs."""
        from datetime import timedelta
        
        cutoff = datetime.now() - timedelta(hours=self.retention_hours)
        
        to_remove = []
        for job_id, job in self.jobs.items():
            if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                if job.completed_at:
                    completed = datetime.fromisoformat(job.completed_at)
                    if completed < cutoff:
                        to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.jobs[job_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old jobs")
    
    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics."""
        stats = {
            "total": len(self.jobs),
            "pending": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
        
        for job in self.jobs.values():
            stats[job.status] = stats.get(job.status, 0) + 1
        
        return stats


# Global job queue
_global_queue: Optional[JobQueue] = None


def get_job_queue() -> JobQueue:
    """Get global job queue instance."""
    global _global_queue
    if _global_queue is None:
        _global_queue = JobQueue()
    return _global_queue
