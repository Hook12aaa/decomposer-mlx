import asyncio
import sqlite3
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    stage: str | None = None
    output_dir: Path | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None


@dataclass
class JobStore:
    job_ttl_seconds: float = 3600.0
    _jobs: dict[str, Job] = field(default_factory=dict)
    _queues: dict[str, list[asyncio.Queue]] = field(default_factory=dict)

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        self._jobs[job.id] = job
        self._queues[job.id] = []
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def update(self, job: Job) -> None:
        self._jobs[job.id] = job

    async def publish(self, job_id: str, event: str, payload: dict) -> None:
        if job_id not in self._queues:
            return
        if event in ("done", "error"):
            job = self._jobs.get(job_id)
            if job is not None:
                job.completed_at = time.time()
        for q in self._queues[job_id]:
            await q.put((event, payload))

    async def subscribe(self, job_id: str) -> AsyncIterator[tuple[str, dict]]:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(job_id, []).append(q)
        try:
            while True:
                event, payload = await q.get()
                yield (event, payload)
                if event in ("done", "error"):
                    return
        finally:
            if q in self._queues.get(job_id, []):
                self._queues[job_id].remove(q)
            if not self._queues.get(job_id):
                job = self._jobs.get(job_id)
                if job is not None and job.completed_at is not None:
                    self._queues.pop(job_id, None)

    def evict_expired(self) -> None:
        now = time.time()
        expired = [
            jid
            for jid, job in self._jobs.items()
            if job.completed_at is not None
            and job.completed_at + self.job_ttl_seconds < now
        ]
        for jid in expired:
            self._jobs.pop(jid, None)
            self._queues.pop(jid, None)


class SqliteJobStore:
    """Persistent job store backed by sqlite3.

    SSE pub/sub remains per-process via in-memory async queues; only the
    Job state is persisted to disk so it survives FastAPI restarts.
    """

    def __init__(self, db_path: Path, job_ttl_seconds: float = 3600.0) -> None:
        self.db_path = Path(db_path)
        self.job_ttl_seconds = job_ttl_seconds
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    stage TEXT,
                    output_dir TEXT,
                    error TEXT,
                    started_at REAL NOT NULL,
                    completed_at REAL
                )"""
            )
            conn.commit()

    def _row_to_job(self, row: tuple) -> Job:
        return Job(
            id=row[0],
            status=row[1],
            stage=row[2],
            output_dir=Path(row[3]) if row[3] else None,
            error=row[4],
            started_at=row[5],
            completed_at=row[6],
        )

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], started_at=time.time())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO jobs (id, status, started_at) VALUES (?, ?, ?)",
                (job.id, job.status, job.started_at),
            )
            conn.commit()
        self._queues[job.id] = []
        return job

    def get(self, job_id: str) -> Job | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT id, status, stage, output_dir, error, started_at, completed_at "
                "FROM jobs WHERE id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    def _update(self, job: Job) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, stage = ?, output_dir = ?, error = ?, "
                "completed_at = ? WHERE id = ?",
                (
                    job.status,
                    job.stage,
                    str(job.output_dir) if job.output_dir is not None else None,
                    job.error,
                    job.completed_at,
                    job.id,
                ),
            )
            conn.commit()

    def update(self, job: Job) -> None:
        self._update(job)

    async def publish(self, job_id: str, event: str, payload: dict) -> None:
        job = self.get(job_id)
        if job is None:
            return
        if event in ("done", "error"):
            job.status = "done" if event == "done" else "error"
            job.completed_at = time.time()
            if event == "error":
                msg = payload.get("message")
                if isinstance(msg, str):
                    job.error = msg
            self._update(job)
        if job_id not in self._queues:
            return
        for q in self._queues[job_id]:
            await q.put((event, payload))

    async def subscribe(self, job_id: str) -> AsyncIterator[tuple[str, dict]]:
        q: asyncio.Queue = asyncio.Queue()
        self._queues.setdefault(job_id, []).append(q)
        try:
            while True:
                event, payload = await q.get()
                yield (event, payload)
                if event in ("done", "error"):
                    return
        finally:
            if q in self._queues.get(job_id, []):
                self._queues[job_id].remove(q)
            if not self._queues.get(job_id):
                job = self.get(job_id)
                if job is not None and job.completed_at is not None:
                    self._queues.pop(job_id, None)

    def evict_expired(self) -> None:
        cutoff = time.time() - self.job_ttl_seconds
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM jobs WHERE completed_at IS NOT NULL AND completed_at < ?",
                (cutoff,),
            )
            conn.commit()
        with sqlite3.connect(self.db_path) as conn:
            existing = {
                row[0] for row in conn.execute("SELECT id FROM jobs").fetchall()
            }
        for jid in list(self._queues.keys()):
            if jid not in existing:
                self._queues.pop(jid, None)
