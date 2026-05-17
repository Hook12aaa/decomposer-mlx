import asyncio

import pytest

from decomposer.web.jobs import JobStore


@pytest.mark.asyncio
async def test_create_job_returns_unique_id():
    store = JobStore()
    a = store.create()
    b = store.create()
    assert a.id != b.id
    assert a.status == "queued"


@pytest.mark.asyncio
async def test_subscribe_receives_events_published_after_subscribe():
    store = JobStore()
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


@pytest.mark.asyncio
async def test_job_queue_is_cleaned_when_subscriber_disconnects_after_done():
    store = JobStore()
    job = store.create()

    async def consumer():
        async for _ in store.subscribe(job.id):
            pass

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)
    await store.publish(job.id, "done", {})
    await asyncio.wait_for(task, timeout=1.0)
    assert store._queues.get(job.id, None) is None or store._queues[job.id] == []


@pytest.mark.asyncio
async def test_completed_jobs_evicted_after_ttl():
    store = JobStore(job_ttl_seconds=0.1)
    job = store.create()
    await store.publish(job.id, "done", {})
    await asyncio.sleep(0.15)
    store.evict_expired()
    assert store.get(job.id) is None


def test_running_jobs_not_evicted():
    store = JobStore(job_ttl_seconds=0.0)
    job = store.create()
    store.evict_expired()
    assert store.get(job.id) is not None
