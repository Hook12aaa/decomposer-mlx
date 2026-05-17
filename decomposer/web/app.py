import asyncio
import io
import json
import logging
import zipfile

import torch
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from decomposer.config import Settings, get_settings
from decomposer.core.backend import InferenceBackend
from decomposer.core.perfetto import report_to_json, to_perfetto
from decomposer.core.xray import Tracer
from decomposer.logging_setup import configure_logging
from decomposer.web.jobs import JobStore, SqliteJobStore

logger = logging.getLogger(__name__)

ASSETS = Path(__file__).resolve().parent.parent / "test_assets"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
MAX_ZIP_BYTES = 500 * 1024 * 1024


class JobRequest(BaseModel):
    sample: str
    layers: int = 6
    resolution: int = 640
    steps: int = 8


def create_app(
    backend: InferenceBackend | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings if settings is not None else get_settings()
    global MAX_ZIP_BYTES
    MAX_ZIP_BYTES = settings.max_zip_bytes
    store: JobStore | SqliteJobStore = (
        SqliteJobStore(
            settings.job_store_db_path, job_ttl_seconds=settings.job_ttl_seconds
        )
        if settings.job_store_db_path is not None
        else JobStore(job_ttl_seconds=settings.job_ttl_seconds)
    )
    inference_lock = asyncio.Lock()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    runs_root = settings.runs_dir.resolve()
    inference_timeout_seconds = settings.inference_timeout_seconds

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal backend
        configure_logging(settings)
        logger.info("web app startup")
        if backend is None:
            if settings.use_fake_backend:
                from decomposer.core.backend import FakeBackend

                logger.info("instantiating FakeBackend")
                backend = FakeBackend(latency_ms=200)
            else:
                from decomposer.core.mps_backend import MpsBackend

                logger.info("instantiating MpsBackend")
                backend = MpsBackend(settings=settings)
        yield
        logger.info("web app shutdown")

    app = FastAPI(lifespan=lifespan)
    app.state.store = store
    app.state.settings = settings

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        return templates.TemplateResponse(request, "index.html", {})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        import torch
        if not torch.backends.mps.is_available():
            return JSONResponse({"status": "mps_unavailable"}, status_code=503)
        return JSONResponse({"status": "ready"}, status_code=200)

    @app.post("/jobs")
    async def post_job(req: JobRequest) -> dict:
        img_path = ASSETS / req.sample
        if not img_path.exists():
            raise HTTPException(404, f"sample not found: {req.sample}")
        job = store.create()
        logger.info(
            "job created id=%s sample=%s layers=%d resolution=%d steps=%d",
            job.id, req.sample, req.layers, req.resolution, req.steps,
        )
        asyncio.create_task(_run_job(
            job.id, img_path, req, store, backend, inference_lock,
            runs_root=runs_root, inference_timeout_seconds=inference_timeout_seconds,
        ))
        return {"job_id": job.id, "stream": f"/sse/{job.id}"}

    @app.get("/sse/{job_id}")
    async def sse(job_id: str) -> EventSourceResponse:
        if store.get(job_id) is None:
            raise HTTPException(404)

        async def gen():
            async for event, payload in store.subscribe(job_id):
                yield {"event": event, "data": json.dumps(payload)}

        return EventSourceResponse(gen())

    @app.get("/jobs/{job_id}/zip")
    async def get_zip(job_id: str) -> StreamingResponse:
        job = store.get(job_id)
        if job is None or job.status != "done" or job.output_dir is None:
            raise HTTPException(404)

        out_dir = job.output_dir.resolve()
        if not out_dir.is_relative_to(runs_root):
            raise HTTPException(403, "output_dir outside runs root")

        total = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
        if total > MAX_ZIP_BYTES:
            raise HTTPException(413, "result too large")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(out_dir.iterdir()):
                if f.is_file():
                    z.write(f, arcname=f.name)
        buf.seek(0)
        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
        )

    return app


async def _run_job(
    job_id: str,
    img_path: Path,
    req: JobRequest,
    store: JobStore | SqliteJobStore,
    backend: InferenceBackend,
    lock: asyncio.Lock,
    runs_root: Path | None = None,
    inference_timeout_seconds: float = 600.0,
) -> None:
    job = store.get(job_id)
    if job is None:
        return
    base = runs_root if runs_root is not None else Path("runs").resolve()
    out_dir = base / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    job.output_dir = out_dir
    store.update(job)
    tracer = Tracer(run_id=job_id)

    async def fwd(event: str, payload: dict) -> None:
        await store.publish(job_id, event, payload)

    def listener(event: str, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(fwd(event, payload), loop)

    loop = asyncio.get_running_loop()
    tracer.subscribe(listener)

    job.status = "running"
    store.update(job)
    try:
        async with lock:
            img = Image.open(img_path).convert("RGB")
            layers = await asyncio.wait_for(
                asyncio.to_thread(
                    backend.decompose, img, req.layers, req.resolution, req.steps, None, tracer
                ),
                timeout=inference_timeout_seconds,
            )
        for i, layer in enumerate(layers):
            layer.save(out_dir / f"layer_{i}.png")
        rep = tracer.report()
        (out_dir / "trace.json").write_text(report_to_json(rep))
        (out_dir / "trace.perfetto.json").write_text(json.dumps(to_perfetto(rep)))
        job.status = "done"
        logger.info("job done id=%s layers=%d", job_id, len(layers))
        await store.publish(job_id, "done", {"layers": len(layers)})
    except TimeoutError:
        job.status = "error"
        job.error = f"inference exceeded {inference_timeout_seconds}s wall-time budget"
        logger.error("job timeout id=%s budget=%.1fs", job_id, inference_timeout_seconds)
        await store.publish(job_id, "error", {"message": job.error})
    except torch.OutOfMemoryError as e:
        job.status = "error"
        job.error = f"OOM: {e}"
        logger.exception("job failed id=%s kind=OOM", job_id)
        await store.publish(job_id, "error", {"message": job.error, "kind": "OOM"})
    except (RuntimeError, MemoryError) as e:
        msg = str(e)
        is_oom = (
            isinstance(e, MemoryError)
            or "out of memory" in msg.lower()
            or "MPS backend out of memory" in msg
        )
        kind = "OOM" if is_oom else "RuntimeError"
        job.status = "error"
        job.error = f"{kind}: {msg}"
        logger.exception("job failed id=%s kind=%s", job_id, kind)
        await store.publish(job_id, "error", {"message": job.error, "kind": kind})
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        logger.exception("inference failed for job %s", job_id)
        await store.publish(job_id, "error", {"message": str(e)})


app = create_app()
