# Decomposer Enterprise Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute. Tasks group fixes by theme.

**Goal:** Take the decomposer v1 from "works end-to-end" to "credible enterprise-internal tool" by addressing every Critical + Important finding from the 2026-05-18 audit, plus the deployment/logging/persistence gaps.

**Source of truth:** Audit findings from the requesting-code-review pass (2026-05-18). 6 Critical + 15 Important + Minor items.

**Tech additions:** `pydantic-settings`, stdlib `logging`, `hashlib` for GGUF verification, `aiosqlite` for persistent JobStore, Docker.

---

## Task Group A — Critical safety (BLOCKERS)

These must land first; everything else builds on them.

### Task A1: Restore the residency invariant — VAE-encode before DiT load

**Files:**
- Modify: `decomposer/core/mps_backend.py`
- Modify: `tests/test_mps_backend.py`

The VAE currently bypasses `ResidencyManager` because `pipeline.prepare_latents` calls `vae.encode(image)`. Fix: pre-encode the input image to a latent ourselves under a proper `load_vae` stage, then pass `latents=` to `pipe(...)` so the pipeline skips its own encode.

- [ ] **Step 1: Write failing test for residency invariant during decompose**
```python
@pytest.mark.mps_required
def test_decompose_never_holds_two_modules_simultaneously():
    backend = MpsBackend()
    img = Image.new("RGB", (64, 64), (200, 50, 50))
    t = Tracer(run_id="r-resid")
    backend.decompose(img, layers=3, resolution=640, steps=4, tracer=t)
    expected_order = [
        "load_text_encoder", "encode_prompt", "free_text_encoder",
        "load_vae", "encode_image_to_latent", "free_vae",
        "load_dit", "denoise_loop", "free_dit",
        "load_vae", "decode_layers", "free_vae",
    ]
    seen = [s.name for s in t.report().stages]
    for name in expected_order:
        assert name in seen, f"missing stage: {name}"
```

- [ ] **Step 2: Refactor `_denoise` and add `_encode_image`**

Add a new phase `_encode_image_to_latent(image, *, tracer) -> torch.Tensor` that:
- Loads VAE via `self.residency.load("vae", ...)`
- Runs `vae.encode(...)` to produce input latent
- Frees VAE via `self.residency.free()`
- Returns the latent (on CPU)

Then `_denoise` should:
- Take the pre-encoded image latent as a parameter
- Load only the DiT (no VAE)
- Pass `latents=image_latent.to(device)` to `pipe(...)` to skip pipeline's own encode
- Free DiT cleanly
- Return final denoised latents

Then `_decode` should:
- Re-load VAE under residency
- Decode N RGBA layers
- Free VAE

- [ ] **Step 3: Verify**
Run: `uv run pytest tests/test_mps_backend.py -v -m mps_required` — both tests pass.

- [ ] **Step 4: Commit** `"Restore at-most-one-MPS-module invariant via pre-encode"`

### Task A2: Replace `UnboundLocalError` catch with proper pipeline override

**Files:**
- Modify: `decomposer/core/mps_backend.py`

The `try/except UnboundLocalError` workaround for diffusers' `output_type="latent"` bug is a CLAUDE-rule violation.

- [ ] **Step 1: Subclass `QwenImageLayeredPipeline` with a `__call__` override**

Add to `mps_backend.py` (or new `_pipeline.py`):
```python
class _LatentOutputPipeline(QwenImageLayeredPipeline):
    """Override __call__ to return latents cleanly without diffusers' UnboundLocalError bug."""
    def _denoise_only(self, **kwargs) -> torch.Tensor:
        captured: dict[str, torch.Tensor] = {}
        original_cb = kwargs.pop("callback_on_step_end", None)
        def _cb(p, i, t, cbk):
            if "latents" in cbk:
                captured["latents"] = cbk["latents"]
            if original_cb is not None:
                return original_cb(p, i, t, cbk)
            return cbk
        kwargs["output_type"] = "latent"
        kwargs["callback_on_step_end"] = _cb
        try:
            super().__call__(**kwargs)
        except UnboundLocalError:
            pass
        if "latents" not in captured:
            raise RuntimeError("denoise loop produced no captured latents")
        return captured["latents"]
```

Then `_denoise` calls `pipe._denoise_only(...)` instead of `pipe(...)`. The `try/except` is now scoped to a single line, documented as the known diffusers bug, and surfaces clearly if anything else changes.

- [ ] **Step 2: Pin upstream bug context**

Add a brief comment near the `try` block referencing the diffusers commit / line where the bug lives, so future maintainers can check if it's been fixed upstream.

- [ ] **Step 3: Verify** `uv run pytest tests/test_mps_backend.py -v -m mps_required` still passes.

- [ ] **Step 4: Commit** `"Encapsulate diffusers UnboundLocalError workaround in pipeline subclass"`

### Task A3: Pin diffusers to a specific commit

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Find current diffusers commit**
```bash
uv run python -c "import diffusers; print(diffusers.__version__)"
ls /Users/hook/Documents/coding/python/consulting/content/.venv/lib/python3.14/site-packages/diffusers/*.dist-info* | head -1
```

- [ ] **Step 2: Pin to that commit in `pyproject.toml`**
Replace `"diffusers @ git+https://github.com/huggingface/diffusers.git"` with `"diffusers @ git+https://github.com/huggingface/diffusers.git@<SHA>"`.

- [ ] **Step 3: Refresh lock**
`uv lock` — confirm reproducible install.

- [ ] **Step 4: Commit** `"Pin diffusers to a specific commit for reproducible builds"`

### Task A4: JobStore eviction + TTL

**Files:**
- Modify: `decomposer/web/jobs.py`
- Modify: `tests/test_web_jobs.py`

- [ ] **Step 1: Failing test**
```python
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
    assert job.id not in store._queues or store._queues[job.id] == []

@pytest.mark.asyncio
async def test_completed_jobs_evicted_after_ttl():
    store = JobStore(job_ttl_seconds=0.1)
    job = store.create()
    await store.publish(job.id, "done", {})
    await asyncio.sleep(0.2)
    store.evict_expired()
    assert store.get(job.id) is None
```

- [ ] **Step 2: Implement** `JobStore(job_ttl_seconds=3600.0)` constructor, `evict_expired()` method, and `_queues[job_id]` removal when the queue empties after a terminal event.

- [ ] **Step 3: Commit** `"JobStore: evict completed jobs after TTL, clean queues on done"`

### Task A5: ZIP route path-containment + size cap

**Files:**
- Modify: `decomposer/web/app.py`
- Modify: `tests/test_web_app.py`

- [ ] **Step 1: Failing test**
```python
@pytest.mark.asyncio
async def test_zip_route_rejects_out_of_root_output_dir(client_factory):
    async with client_factory() as client:
        # Manually set a malicious output_dir on a job
        app = client._transport.app
        job = app.state.store.create()
        job.status = "done"
        job.output_dir = Path("/etc")
        r = await client.get(f"/jobs/{job.id}/zip")
        assert r.status_code == 403
```

- [ ] **Step 2: Add containment + size checks**

In `get_zip`:
```python
runs_root = Path("runs").resolve()
out_dir = job.output_dir.resolve()
if not out_dir.is_relative_to(runs_root):
    raise HTTPException(403, "output_dir outside runs root")
total_bytes = sum(f.stat().st_size for f in out_dir.iterdir() if f.is_file())
if total_bytes > MAX_ZIP_BYTES:
    raise HTTPException(413, "result too large")
```

- [ ] **Step 3: Commit** `"Web: containment + size cap on ZIP responses"`

### Task A6: Vectorize `_unpack_q8_0_to_tensors`

**Files:**
- Modify: `decomposer/core/gguf_loader.py`
- Add: `tests/test_gguf_loader.py` — equivalence test vs. current loop implementation
- Delete unused `dequantize_q8_0` (or move to a `_debug.py` if needed for inspection).

- [ ] **Step 1: Equivalence test** (must pass before and after the rewrite)
```python
def test_vectorized_unpack_matches_loop():
    rng = np.random.default_rng(99)
    weight = rng.standard_normal((256, 128)).astype(np.float32) * 0.1
    packed = _pack_q8_0(weight)
    quants_new, scales_new = _unpack_q8_0_to_tensors(packed, (256, 128))
    # Reference: loop-style unpack
    n_elements = 256 * 128
    n_blocks = n_elements // 32
    arr = np.frombuffer(packed, dtype=np.uint8)
    scales_ref = np.empty(n_blocks, dtype=np.float16)
    quants_ref = np.empty((n_blocks, 32), dtype=np.int8)
    for b in range(n_blocks):
        off = b * 34
        scales_ref[b] = np.frombuffer(arr[off:off+2].tobytes(), dtype=np.float16)[0]
        quants_ref[b] = np.frombuffer(arr[off+2:off+34].tobytes(), dtype=np.int8)
    assert torch.allclose(quants_new.to(torch.int64), torch.from_numpy(quants_ref).to(torch.int64))
    assert torch.allclose(scales_new.to(torch.float32), torch.from_numpy(scales_ref).to(torch.float32))
```

- [ ] **Step 2: Vectorized implementation**

```python
def _unpack_q8_0_to_tensors(packed: bytes, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    n_elements = int(np.prod(shape))
    n_blocks = n_elements // Q8_BLOCK_SIZE
    arr = np.frombuffer(packed, dtype=np.uint8).reshape(n_blocks, Q8_BLOCK_BYTES)
    scales_np = arr[:, :2].copy().view(np.float16).reshape(n_blocks)
    quants_np = arr[:, 2:].copy().view(np.int8)
    return torch.from_numpy(quants_np), torch.from_numpy(scales_np)
```

- [ ] **Step 3: Commit** `"GgufLinear: vectorize Q8 unpack (10-100x faster cold-start)"`

---

## Task Group B — Configuration & secrets

### Task B1: `Settings` class with pydantic-settings

**Files:**
- Create: `decomposer/config.py`
- Modify: `decomposer/core/mps_backend.py`, `decomposer/cli.py`, `decomposer/web/app.py`
- Create: `tests/test_config.py`
- Add: `.env.example`

- [ ] **Step 1: Add `pydantic-settings>=2.0` to `pyproject.toml`**

- [ ] **Step 2: Create `decomposer/config.py`**
```python
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DECOMPOSER_")

    hf_repo: str = "Qwen/Qwen-Image-Layered"
    text_encoder_repo: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    gguf_repo: str = "unsloth/Qwen-Image-Layered-GGUF"
    gguf_file: str = "qwen-image-layered-Q8_0.gguf"

    runs_dir: Path = Path("runs")
    max_zip_bytes: int = 500 * 1024 * 1024
    inference_timeout_seconds: float = 600.0
    job_ttl_seconds: float = 3600.0

    hf_token: str | None = Field(default=None, repr=False)


def get_settings() -> Settings:
    return Settings()
```

- [ ] **Step 3: Replace all module-level constants with `Settings()` lookups.** Inject via constructor where possible (`MpsBackend(settings=...)`, `create_app(settings=...)`).

- [ ] **Step 4: Test that env overrides work**
```python
def test_settings_respects_env_override(monkeypatch):
    monkeypatch.setenv("DECOMPOSER_INFERENCE_TIMEOUT_SECONDS", "30.0")
    s = Settings()
    assert s.inference_timeout_seconds == 30.0
```

- [ ] **Step 5: Add `.env.example`** documenting every setting.

- [ ] **Step 6: Commit** `"Add Settings config layer (pydantic-settings + .env)"`

### Task B2: HF auth flow in `doctor`

**Files:**
- Modify: `decomposer/cli.py`

- [ ] **Step 1: Add HF auth check to `doctor` BEFORE attempting any model load**
```python
def _check_hf_auth(settings: Settings) -> bool:
    from huggingface_hub import HfApi
    try:
        api = HfApi(token=settings.hf_token)
        api.repo_info(settings.hf_repo)
        return True
    except Exception as e:
        console.print(f"[red]HF auth failed for {settings.hf_repo}[/red]: {e}")
        console.print("To fix:")
        console.print(f"  1. Visit https://huggingface.co/{settings.hf_repo} and accept the license")
        console.print(f"  2. Run: huggingface-cli login")
        console.print(f"  3. Or set DECOMPOSER_HF_TOKEN in .env")
        return False
```

Add it to `doctor` before the FakeBackend / real backend dispatch.

- [ ] **Step 2: Commit** `"doctor: pre-check HF auth with clear remediation on failure"`

---

## Task Group C — Observability hardening

### Task C1: Tracer thread-safety — replace global warnings mutation

**Files:**
- Modify: `decomposer/core/xray.py`
- Modify: `tests/test_xray.py`

- [ ] **Step 1: Replace `warnings.showwarning` patching with `warnings.catch_warnings` context manager**

Use a per-Tracer context manager that captures warnings without mutating global state:
```python
@contextmanager
def stage(self, name: str, **extras):
    rec = self._begin(name, extras)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            yield
        for w in caught:
            text = str(w.message)
            if "fell back to CPU" in text or ("MPS:" in text and "fallback" in text):
                rec.device_fallbacks += 1
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    finally:
        self._end(rec)
```

This removes the `_install_fallback_hook` global mutation entirely.

- [ ] **Step 2: Verify** all existing `xray` tests still pass.

- [ ] **Step 3: Add a concurrency test**
```python
def test_two_tracers_can_run_concurrently_in_threads():
    import threading
    results = {}
    def run(tracer_id):
        t = Tracer(run_id=tracer_id)
        with t.stage("a"):
            warnings.warn("aten::op fell back to CPU", UserWarning)
        results[tracer_id] = t.report().total_fallbacks()
    threads = [threading.Thread(target=run, args=(f"t{i}",)) for i in range(4)]
    for th in threads: th.start()
    for th in threads: th.join()
    assert all(c == 1 for c in results.values())
```

- [ ] **Step 4: Commit** `"Tracer: thread-safe fallback capture via catch_warnings"`

### Task C2: Stdlib logging migration

**Files:**
- Create: `decomposer/logging_setup.py`
- Modify: All modules using `print()` or `console.print()` for non-UI output

- [ ] **Step 1: Create `decomposer/logging_setup.py`**
```python
import logging
import sys
from pathlib import Path

def configure_logging(settings: "Settings") -> None:
    handlers = [logging.StreamHandler(sys.stderr)]
    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(settings.runs_dir / "decomposer.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=handlers,
    )
```

- [ ] **Step 2: Convert non-UI output** to `logger = logging.getLogger(__name__)` + `logger.info(...)`. Keep `rich` for the CLI tables (UI) and progress.

- [ ] **Step 3: Wire into FastAPI lifespan and CLI entry.**

- [ ] **Step 4: Commit** `"Add stdlib logging configuration; wire into web + CLI"`

### Task C3: Tracer `_emit` outside timed regions

**Files:**
- Modify: `decomposer/core/xray.py`

- [ ] **Step 1**: Move the `_emit("stage_started", ...)` call so it happens AFTER `_start_ts` is recorded, but the listener invocations themselves don't count against `wall_ms`. The cleanest pattern: snapshot the listener events into a queue and emit them async, OR record times BEFORE the emit.

- [ ] **Step 2**: Add a test that asserts a slow listener doesn't inflate `wall_ms`:
```python
def test_slow_listener_does_not_inflate_wall_ms():
    t = Tracer(run_id="r")
    t.subscribe(lambda e, p: time.sleep(0.05))
    with t.stage("instant"):
        pass
    assert t.report().stages[0].wall_ms < 30.0
```

- [ ] **Step 3: Commit** `"Tracer: listener overhead no longer counts toward wall_ms"`

---

## Task Group D — Test coverage hardening (CI gate)

### Task D1: CPU-mode contract tests

**Files:**
- Create: `tests/test_residency_contract.py`
- Create: `tests/test_q8_memory_contract.py`

These run in CI (not `mps_required`) and verify the design invariants with synthetic tensors.

- [ ] **Step 1: Residency contract test (CPU)**

Use small `nn.Linear` modules + a `cpu` device. Verify:
- After `load("text", ...)` then `load("dit", ...)`, the text module's tensors are deallocated (use `weakref` to verify)
- Two sequential loads never see two modules alive at once

- [ ] **Step 2: Q8 memory-shape contract test (CPU)**

Create a small `GgufLinear` from synthetic packed bytes. Assert `quants` is int8, `scales` is fp16, no `weight` buffer.

- [ ] **Step 3: Register `mps_required` marker properly in pyproject**
```toml
[tool.pytest.ini_options]
markers = ["mps_required: requires Apple Silicon with MPS available"]
```

- [ ] **Step 4: Commit** `"Add CPU-mode contract tests for residency and Q8 memory"`

---

## Task Group E — Deployment & operations

### Task E1: `/healthz` and `/readyz` endpoints

**Files:**
- Modify: `decomposer/web/app.py`
- Modify: `tests/test_web_app.py`

- [ ] **Step 1: Add health routes**
```python
@app.get("/healthz")
async def healthz():
    return {"status": "ok"}

@app.get("/readyz")
async def readyz():
    # ready when MpsBackend can answer basic probe
    if not torch.backends.mps.is_available():
        return JSONResponse({"status": "mps_unavailable"}, status_code=503)
    return {"status": "ready"}
```

- [ ] **Step 2: Test** both return 200 in the FakeBackend test app.

- [ ] **Step 3: Commit** `"Add /healthz and /readyz endpoints"`

### Task E2: Dockerfile (Linux + CPU-only fallback for CI)

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`

- [ ] **Step 1: Multi-stage Dockerfile** that:
- Stage 1: installs uv, runs `uv sync --extra dev`
- Stage 2: copies app, sets entrypoint to `uvicorn decomposer.web.app:app`

Note: this is Linux-only and CPU-only — MPS inference isn't available in Docker. The container is for the FakeBackend dev experience and CI.

- [ ] **Step 2: Document MPS limitations in README.**

- [ ] **Step 3: Commit** `"Add Dockerfile (CPU-only / dev path)"`

### Task E3: Persistent JobStore (SQLite via aiosqlite)

**Files:**
- Modify: `decomposer/web/jobs.py`
- Add `aiosqlite>=0.20` to deps

- [ ] **Step 1: Add an optional persistence layer.** Keep the in-memory `JobStore` as the default but allow a `SqliteJobStore(db_path)` that survives restarts.

- [ ] **Step 2: Schema**
```sql
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    stage TEXT,
    output_dir TEXT,
    error TEXT,
    started_at REAL,
    completed_at REAL
);
```

- [ ] **Step 3: Tests** verifying persistence across `Store.__init__()` calls.

- [ ] **Step 4: Commit** `"Add SqliteJobStore for persistent job state"`

### Task E4: GGUF checksum verification

**Files:**
- Modify: `decomposer/core/gguf_pipeline.py`

- [ ] **Step 1: After download, compute SHA256 of the GGUF file and compare to a known-good hash** stored in `decomposer/config.py` (or a JSON manifest). Refuse to load on mismatch with a clear error.

- [ ] **Step 2: Test** with a tampered tiny mock file.

- [ ] **Step 3: Commit** `"GGUF loader: verify SHA256 before loading"`

---

## Task Group F — Polish

### Task F1: Add `started_at` to `Job`
Trivial spec drift fix in `web/jobs.py`. Use `time.time()` in `create()`.

### Task F2: Narrow exception handling in `_run_job`
Replace substring "out of memory" matching with `torch.cuda.OutOfMemoryError` / `torch.mps.OutOfMemoryError` (when available) — and a typed `DecomposerError` hierarchy.

### Task F3: README expansion
Add sections for: HF auth, first-run timing, common errors, operational notes, dev vs. prod.

### Task F4: Remove `cli.py:39` package-dir write
Move the on-demand `tiny_smoke_test.png` generation into `settings.runs_dir`, not the package install dir.

---

## Build order

1. Group A (Critical safety) — must land first
2. Group B (Configuration) — unblocks env-driven deployment
3. Group C (Observability) — depends on Settings
4. Group D (Tests) — runs against the new code
5. Group E (Deployment) — depends on health endpoints + persistent store
6. Group F (Polish) — last

Each group is one commit. Group A is sequential (A1 → A2 → A3 → A4 → A5 → A6). Groups B-F can largely parallelize within each group.

## Done criteria

- All Critical issues from audit fixed
- All Important issues from audit fixed
- `/healthz` returns 200; `/readyz` returns 503 on non-MPS hosts
- `uv run pytest -m "not mps_required"` covers residency + Q8 contracts
- Settings via env var works end-to-end (override `DECOMPOSER_INFERENCE_TIMEOUT_SECONDS=30` and observe behavior change)
- Dockerfile builds; container runs FakeBackend demo
- README covers HF auth + first-run + ops
- No `try/except` masks real bugs; no hardcoded constants; diffusers pinned
