import asyncio
from pathlib import Path

import pytest

from decomposer.web.jobs import SqliteJobStore


@pytest.mark.asyncio
async def test_sqlite_persists_across_instances(tmp_path: Path):
    db = tmp_path / "jobs.db"

    store1 = SqliteJobStore(db)
    job = store1.create()
    await store1.publish(job.id, "done", {})

    store2 = SqliteJobStore(db)
    retrieved = store2.get(job.id)
    assert retrieved is not None
    assert retrieved.id == job.id
    assert retrieved.status == "done"
    assert retrieved.completed_at is not None
    assert retrieved.started_at > 0


@pytest.mark.asyncio
async def test_sqlite_persists_error_message(tmp_path: Path):
    db = tmp_path / "jobs.db"
    store = SqliteJobStore(db)
    job = store.create()
    await store.publish(job.id, "error", {"message": "boom"})

    other = SqliteJobStore(db)
    retrieved = other.get(job.id)
    assert retrieved is not None
    assert retrieved.status == "error"
    assert retrieved.error == "boom"


def test_sqlite_update_persists_output_dir(tmp_path: Path):
    db = tmp_path / "jobs.db"
    store = SqliteJobStore(db)
    job = store.create()
    job.status = "running"
    job.output_dir = tmp_path / "runs" / job.id
    store.update(job)

    other = SqliteJobStore(db)
    retrieved = other.get(job.id)
    assert retrieved is not None
    assert retrieved.status == "running"
    assert retrieved.output_dir == tmp_path / "runs" / job.id


def test_sqlite_evict_removes_old_completed_jobs(tmp_path: Path):
    db = tmp_path / "jobs2.db"
    store = SqliteJobStore(db, job_ttl_seconds=0.0)
    job = store.create()
    asyncio.run(store.publish(job.id, "done", {}))
    store.evict_expired()
    assert store.get(job.id) is None


def test_sqlite_evict_keeps_running_jobs(tmp_path: Path):
    db = tmp_path / "jobs3.db"
    store = SqliteJobStore(db, job_ttl_seconds=0.0)
    job = store.create()
    store.evict_expired()
    assert store.get(job.id) is not None


@pytest.mark.asyncio
async def test_sqlite_subscribe_receives_events(tmp_path: Path):
    db = tmp_path / "jobs4.db"
    store = SqliteJobStore(db)
    job = store.create()
    received: list[tuple[str, dict]] = []

    async def consumer():
        async for ev in store.subscribe(job.id):
            received.append(ev)
            if ev[0] == "done":
                return

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    await store.publish(job.id, "stage_started", {"name": "load_dit"})
    await store.publish(job.id, "done", {})
    await asyncio.wait_for(task, timeout=1.0)
    assert received[0] == ("stage_started", {"name": "load_dit"})
    assert received[-1] == ("done", {})
