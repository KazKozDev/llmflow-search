"""Unit tests for the background job queue."""

import asyncio
import unittest
from datetime import datetime, timedelta

from core.background_jobs import JobQueue, JobStatus


class BackgroundJobQueueTests(unittest.IsolatedAsyncioTestCase):
    """Verify job lifecycle transitions and queue bookkeeping."""

    async def test_completed_job_updates_status_and_stats(self):
        """A successful worker should mark the job completed."""
        queue = JobQueue(max_concurrent=1)

        async def worker(job):
            queue.update_progress(job.id, 50)
            return "done"

        job_id = await queue.submit("query", worker)

        while queue.running_tasks:
            await asyncio.sleep(0.01)

        job = queue.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, JobStatus.COMPLETED)
        self.assertEqual(job.result, "done")
        self.assertEqual(job.progress, 100)
        self.assertEqual(queue.get_stats()["completed"], 1)

    async def test_failed_job_captures_error(self):
        """Worker exceptions should mark the job as failed."""
        queue = JobQueue(max_concurrent=1)

        async def worker(_job):
            raise RuntimeError("worker failed")

        job_id = await queue.submit("query", worker)

        while queue.running_tasks:
            await asyncio.sleep(0.01)

        job = queue.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, JobStatus.FAILED)
        self.assertIn("worker failed", job.error)

    async def test_cancel_running_job_marks_job_cancelled(self):
        """Cancelling a running task should update the final status."""
        queue = JobQueue(max_concurrent=1)
        started = asyncio.Event()

        async def worker(_job):
            started.set()
            await asyncio.sleep(10)

        job_id = await queue.submit("query", worker)
        await started.wait()

        cancelled = await queue.cancel_job(job_id)
        self.assertTrue(cancelled)

        while queue.running_tasks:
            await asyncio.sleep(0.01)

        job = queue.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, JobStatus.CANCELLED)

    async def test_cleanup_old_jobs_removes_expired_terminal_jobs(self):
        """Cleanup should remove completed jobs older than retention."""
        queue = JobQueue(max_concurrent=1, retention_hours=1)

        async def worker(_job):
            return "done"

        job_id = await queue.submit("query", worker)

        while queue.running_tasks:
            await asyncio.sleep(0.01)

        job = queue.get_job(job_id)
        self.assertIsNotNone(job)
        job.completed_at = (datetime.now() - timedelta(hours=2)).isoformat()

        await queue.cleanup_old_jobs()

        self.assertIsNone(queue.get_job(job_id))


if __name__ == "__main__":
    unittest.main()
