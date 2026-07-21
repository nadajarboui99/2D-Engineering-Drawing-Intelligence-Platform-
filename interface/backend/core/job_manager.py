"""
Central job manager.
Every long-running evaluation runs as a background job.
Frontend polls GET /jobs/{job_id} to get live status + logs.
"""

import uuid
import asyncio
from datetime import datetime
from typing import Callable
from enum import Enum

class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    DONE     = "done"
    FAILED   = "failed"

class Job:
    def __init__(self, job_id: str, name: str):
        self.job_id    = job_id
        self.name      = name
        self.status    = JobStatus.PENDING
        self.logs      = []
        self.result    = None
        self.error     = None
        self.created_at = datetime.utcnow().isoformat()
        self.finished_at = None

    def log(self, message: str):
        self.logs.append({"ts": datetime.utcnow().isoformat(), "msg": message})

    def to_dict(self):
        return {
            "job_id":      self.job_id,
            "name":        self.name,
            "status":      self.status,
            "logs":        self.logs,
            "result":      self.result,
            "error":       self.error,
            "created_at":  self.created_at,
            "finished_at": self.finished_at,
        }

# In-memory store (replace with Redis for production)
_jobs: dict[str, Job] = {}


def create_job(name: str) -> Job:
    job_id = str(uuid.uuid4())
    job = Job(job_id, name)
    _jobs[job_id] = job
    return job


def get_job(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def list_jobs() -> list[dict]:
    return [j.to_dict() for j in reversed(list(_jobs.values()))]


async def run_job(job: Job, fn: Callable, **kwargs):
    """Run fn(**kwargs) in a thread so it doesn't block the event loop."""
    job.status = JobStatus.RUNNING
    job.log(f"Starting {job.name}...")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, lambda: fn(job=job, **kwargs))
        job.result = result
        job.status = JobStatus.DONE
        job.log("Done.")
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error  = str(e)
        job.log(f"Error: {e}")
    finally:
        from datetime import datetime
        job.finished_at = datetime.utcnow().isoformat()