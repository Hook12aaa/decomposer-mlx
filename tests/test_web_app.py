from pathlib import Path
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from decomposer.core.backend import FakeBackend
from decomposer.web.app import create_app


@pytest.fixture
def client_factory():
    def make():
        app = create_app(backend=FakeBackend(latency_ms=10))
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    return make


@pytest.mark.asyncio
async def test_index_renders(client_factory):
    async with client_factory() as client:
        r = await client.get("/")
        assert r.status_code == 200
        assert "Decomposer" in r.text


@pytest.mark.asyncio
async def test_post_jobs_returns_job_id(client_factory):
    async with client_factory() as client:
        r = await client.post(
            "/jobs",
            json={"sample": "tiny_smoke_test.png", "layers": 3, "resolution": 128, "steps": 4},
        )
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body and "stream" in body


@pytest.mark.asyncio
async def test_zip_route_returns_404_before_done(client_factory):
    async with client_factory() as client:
        r = await client.get("/jobs/nonexistent/zip")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_zip_rejects_output_dir_outside_runs(client_factory):
    async with client_factory() as client:
        app = client._transport.app
        store = app.state.store
        job = store.create()
        job.status = "done"
        job.output_dir = Path("/etc")
        r = await client.get(f"/jobs/{job.id}/zip")
        assert r.status_code == 403


@pytest.mark.asyncio
async def test_healthz_returns_ok(client_factory):
    async with client_factory() as client:
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readyz_reports_status(client_factory):
    async with client_factory() as client:
        r = await client.get("/readyz")
        assert r.status_code in (200, 503)
        assert "status" in r.json()


@pytest.mark.asyncio
async def test_zip_rejects_oversized_output(client_factory, tmp_path):
    async with client_factory() as client:
        app = client._transport.app
        store = app.state.store
        job = store.create()
        job.status = "done"
        runs = Path("runs").resolve() / job.id
        runs.mkdir(parents=True, exist_ok=True)
        big = runs / "big.bin"
        with patch("decomposer.web.app.MAX_ZIP_BYTES", 10):
            big.write_bytes(b"x" * 100)
            job.output_dir = runs
            r = await client.get(f"/jobs/{job.id}/zip")
            assert r.status_code == 413
        big.unlink()
        runs.rmdir()
