# Decomposer — Design Spec

**Date:** 2026-05-17
**Status:** Approved for implementation planning
**Project name:** `decomposer`

## Goal

A local-first pipeline that decomposes a flat marketing image into its constituent visual layers (background, subject, decorative elements, text block, etc.) as separate RGBA PNGs, using the [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) model. v1 ships as a Python web app with a single workflow: pick a bundled sample image, run decomposition, download a ZIP of layer PNGs plus a structured performance trace.

## Non-goals (v1)

- No semantic labeling of layers (logo / headline / CTA classification). Layer assignment from Qwen-Image-Layered is emergent; labeling is a v2 concern.
- No OCR or text-block splitting (headline vs subhead vs body vs CTA). v2.
- No interactive layer manipulation in the browser. v1 is upload-equivalent → ZIP download.
- No user-uploaded images. v1 uses bundled sample images only.
- No multi-user concurrency, no auth, no database. Single-user local tool.
- No cloud inference backend. Mac-local only.

## Hardware target

Apple Silicon, M3 Max with **48 GB unified memory**. This is below the comfortable bf16 floor for Qwen-Image-Layered (~50–55 GB resident at bf16). The design accommodates this through quantization, sequential module residency, and fp16 (skipping bf16 emulation).

## Critical caveats

1. **Layer semantics are emergent, not directed.** The model decides what goes in each layer based on visual grouping; you cannot prompt it to put the logo in layer 2. v1 returns the layers as-is.
2. **Qwen-Image-Layered has no public MPS benchmark** as of writing. The layered VAE may expose MPS op gaps that base Qwen-Image does not. `decomposer doctor` is the first thing built — its purpose is to convert this uncertainty into evidence on day one.
3. **fp8mixed quantization does not run on Apple Silicon.** FP8 requires CUDA Ada/Hopper kernels. v1 uses Q8 GGUF instead.
4. **Multi-minute inference is the steady state.** Even with all optimizations, expect 25–60 s per decomposition at 640px / 8 steps; minutes at 1024px / 50 steps.

## Architecture

```
decomposer/
├── core/
│   ├── backend.py            # InferenceBackend Protocol
│   ├── mps_backend.py        # MpsBackend (the one v1 implementation)
│   ├── residency.py          # ResidencyManager — enforces "one module hot at a time"
│   ├── gguf_loader.py        # Custom Q8 GGUF loader for the DiT
│   ├── pipeline.py           # decompose() — orchestrates the five phases
│   └── xray.py               # Tracer + Report + Perfetto exporter
├── cli.py                    # typer commands: doctor, decompose, diff-traces
├── web/
│   ├── app.py                # FastAPI: lifespan, /, /jobs, /sse/{id}, /jobs/{id}/zip
│   └── templates/index.html  # one page, vanilla HTML + ~20 lines JS for SSE
├── test_assets/              # 2–3 bundled marketing sample images
├── runs/                     # written at runtime: per-run outputs + traces
└── tests/                    # see Testing section
```

### Inference pipeline (five phases)

```
Phase 1: prompt-conditioning
  load Qwen2.5-VL-7B (fp16) → encode (prompt + image) → free
       ↓ memory floor ~2 GB

Phase 2: latent preparation
  noise init at (resolution, resolution), scheduler setup

Phase 3: denoising
  load DiT (Q8 GGUF, fp16) → N steps (8 with Lightning LoRA) → free
       ↓ peak ~25 GB during this phase

Phase 4: layer decoding
  load RGBA-VAE (fp16) → decode N RGBA layers (tiled if 1024px) → free
       ↓ peak ~3 GB during this phase

Phase 5: output
  convert tensors → PIL RGBA → return
```

### `InferenceBackend` contract

```python
class InferenceBackend(Protocol):
    def decompose(
        self,
        image: PIL.Image.Image,
        layers: int,
        resolution: int = 640,
        steps: int = 8,
        seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> list[PIL.Image.Image]: ...
```

`MpsBackend` is the only implementation in v1. The Protocol exists so a `RemoteBackend` (Modal / RunPod / fal.ai) can drop in later without touching CLI or web code.

### Concurrency & resource model

- Model is **not** pre-loaded in FastAPI `lifespan`. Cold-start everything on first request (~3–5 min first click; ~2 GB idle RAM). Subsequent requests reuse nothing; each request re-loads through the five phases.
- `asyncio.Lock` wraps every `decompose()` call. Only one inference runs at a time (matches GPU reality, prevents reentrancy bugs).
- The blocking torch code runs via `asyncio.to_thread`, keeping the FastAPI event loop responsive for SSE and other routes during a multi-minute run.
- `ResidencyManager` enforces the invariant: at most one of `{text_encoder, dit, vae}` lives on MPS at any moment.

## Observability — the X-ray layer

Performance instrumentation is a first-class part of the harness, not added on top. Every stage in the pipeline is named, measured, and reportable.

### Instrumented stages

| Stage | Measured |
|---|---|
| `load_text_encoder` | wall, peak RSS, peak MPS alloc |
| `encode_prompt` | wall, GPU time, token count |
| `free_text_encoder` | RSS delta, MPS alloc delta |
| `load_dit` | wall, peak RSS, peak MPS alloc |
| `prepare_latents` | wall |
| `denoise_loop` | wall, per-step breakdown |
| `denoise_step[i]` | wall, GPU time, latent norm |
| `free_dit` | RSS delta, MPS alloc delta |
| `load_vae` | wall, peak RSS, peak MPS alloc |
| `decode_layers` | wall, per-layer wall, peak MPS alloc |
| `free_vae` | RSS delta |
| `write_outputs` | wall |

### Per-stage metrics

- `wall_ms` — total clock time (Python + GPU)
- `gpu_ms` — MPS-synced GPU time (boundaries via `torch.mps.synchronize()`)
- `rss_peak_mb` — process RSS peak
- `mps_alloc_peak_mb` — `torch.mps.driver_allocated_memory()` peak
- `mps_alloc_delta_mb` — change before/after; catches leaks on `free_*` stages
- `device_fallbacks` — count of ops that silently fell back to CPU (critical MPS perf metric)

### CPU-fallback detection

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set so unsupported ops run instead of crashing. A warning-filter hook logs each fallback into the active stage. The X-ray report surfaces fallback counts prominently — this is the closest analog to per-layer CNN profiling for MMDiT on MPS.

### Tracer API

```python
class Tracer:
    @contextmanager
    def stage(self, name: str, **extras): ...
    def step(self, name: str, **extras): ...
    def annotate(self, **extras): ...
    def report(self) -> Report: ...
```

The Tracer also exposes an event stream consumed by the FastAPI SSE endpoint, so the same instrumentation powers both the CLI live console (rich) and the browser progress UI.

### Outputs per run

Three artifacts written to `runs/<timestamp>/`:

1. Live `rich` table on stdout during the run.
2. `trace.json` — full structured trace, diffable across runs.
3. `trace.perfetto.json` — Chrome trace format, opens in `chrome://tracing` or `ui.perfetto.dev` for waterfall visualization.

### Optimization feedback loop

The X-ray is the proof mechanism for every optimization decision:

| Lever | Metric that proves it worked |
|---|---|
| fp16 vs bf16 | `denoise_step.gpu_ms` |
| Q8 GGUF vs full fp16 | `load_dit.mps_alloc_peak_mb` |
| Lightning LoRA (8 steps vs 50) | `denoise_loop.wall_ms` |
| 640px vs 1024px | `denoise_step.gpu_ms` (≈quadratic) |
| Sequential offload vs naive load | global `mps_alloc_peak_mb` |
| VAE tiling | `decode_layers.mps_alloc_peak_mb` |
| CPU-fallback op fixed | per-stage `device_fallbacks` count |

`decomposer diff-traces <run-A> <run-B>` surfaces these deltas with regressions in red.

### Explicit observability non-goals (v1)

- No Prometheus / OpenTelemetry exporter. Single-process tool; JSON files suffice.
- No always-on intermediate-image dumping. Gated behind `--debug-previews`.

## Inference harness — key decisions

### Quantization: Q8 GGUF, no fallback

Committed. The DiT loads via a custom `GgufLinear` module (port of the ComfyUI-GGUF dequant-on-the-fly pattern). This is the most fragile piece of v1 — `decomposer doctor` validates it before any other implementation work proceeds. If it fails on M3 Max, we stop and revisit the quantization decision with real evidence rather than route around it.

### Precision: fp16, not bf16

M1–M3 emulate bf16 in software (significant perf cliff). All three modules load with `dtype=torch.float16`. Qwen-Image's activation range occasionally overflows fp16 — apply activation scaling on the DiT, per the DrawThings team's published workaround.

### Sampling: Lightning LoRA, 8 steps, CFG 1.0

Default. Cuts inference from ~100 forward passes (50 steps × CFG 4) to 8. Quality cost is acceptable for the decomposition task. CFG 4 + 50 steps is a v2 toggle.

### What the harness does not do in v1

- No CFG (Lightning requires CFG=1).
- No multi-image batching.
- No streaming partial layers.

## CLI surface (typer)

```
decomposer doctor
  → reports: PyTorch+MPS version, available unified memory, Q8 GGUF support
  → runs a 10-step decomp on a 256px test image, prints X-ray summary
  → exit 0 if green; non-zero with diagnosis otherwise

decomposer decompose <image-path> [--layers 6] [--resolution 640] [--steps 8]
                                  [--seed 42] [--out ./out/] [--trace]
  → writes layer_0.png ... layer_N.png, composite.png, trace.json (if --trace)
  → live rich table during run

decomposer diff-traces <run-A> <run-B>
  → side-by-side per-stage wall_ms / mps_alloc_peak_mb deltas
  → regressions highlighted red
```

`doctor` is the most important command and is built first.

## FastAPI surface

```
GET  /                     renders index.html
POST /jobs                 starts job; returns {"job_id": "...", "stream": "/sse/<id>"}
GET  /sse/<job_id>         server-sent events: stage_started, stage_ended, progress, done, error
GET  /jobs/<job_id>/zip    streams ZIP of layer_*.png + trace.json once status=done
```

Lifespan instantiates `MpsBackend()` but loads no model weights (cold-start choice).

### Job state (in-memory)

```python
@dataclass
class Job:
    id: str
    status: Literal["queued", "running", "done", "error"]
    stage: str | None
    started_at: float
    output_dir: Path | None
    error: str | None
    trace: Report | None
```

A `dict[str, Job]` is sufficient. No Redis, SQLite, or job queue in v1.

### Frontend page

Vanilla HTML + ~20 lines inline JS for the `EventSource` SSE subscription. No HTMX, no Alpine, no framework. Page contents:

- Dropdown of bundled sample images
- "Decompose" button
- Live progress area (current stage + per-stage wall_ms ticking in)
- Download link (appears when status=done)
- Link to the run's `trace.perfetto.json`

## Error handling

| Failure | Where | Response |
|---|---|---|
| MPS OOM during load | `ResidencyManager.load()` | Free everything, raise `OutOfMemoryError(module, attempted_mb)`; job → error with diagnostic |
| MPS op fallback storm | tracer detects > threshold fallbacks | Job completes; X-ray report flags red |
| Q8 GGUF dequant error | `GgufLinear.forward` | Bubble up with layer name (smoke-test catch) |
| Inference exceeds wall-time budget | `decompose` timeout (default 600 s, configurable) | Cancel job, return timeout error |
| Anything else | anywhere | Crash; traceback to console. Single-user dev tool. |

## Testing strategy

| Test | Purpose | CI? |
|---|---|---|
| `test_doctor.py` | Runs `decomposer doctor` end-to-end on real MPS | No (`@pytest.mark.mps_required`) |
| `test_residency_manager.py` | Load A, assert MPS alloc up; load B, assert A freed first. Mock Module. | Yes |
| `test_tracer.py` | Stage ordering, nested stages, JSON round-trip, Perfetto export validates | Yes |
| `test_pipeline_mocked.py` | FastAPI POST → SSE → ZIP flow with `FakeBackend` returning dummy RGBA | Yes |
| `test_cli_smoke.py` | Typer invocations with `FakeBackend` | Yes |

Notably absent: golden-image pixel-comparison tests on decomposition output. Diffusion non-determinism makes these fool's errands. Rely on `doctor` for "does it work" and human eyeball for "does it look right."

## Tech stack

| Layer | Choice |
|---|---|
| Language | Python 3.12 |
| Package manager | uv |
| Web | FastAPI + uvicorn |
| Templates | Jinja2 (one file) |
| CLI | typer + rich |
| Inference | diffusers (HEAD) + custom GGUF loader + torch (MPS) |
| Quantization | gguf python lib |
| Observability | bespoke `Tracer` + JSON + Perfetto export |
| Testing | pytest + pytest-asyncio |
| Lint/format | ruff |

## Build order

1. `decomposer doctor` (smoke test) — proves MPS + Q8 GGUF + layered pipeline works on this hardware.
2. `Tracer` + `Report` + Perfetto export — needed to read what doctor finds.
3. `ResidencyManager` + `MpsBackend.decompose()` — the core five-phase pipeline.
4. `decomposer decompose` CLI command.
5. FastAPI app + SSE + ZIP streaming.
6. `decomposer diff-traces`.

Step 1 is a gate. If it fails, stop and revisit quantization before proceeding.

## Open risks

- **Q8 GGUF custom loader on MPS**: highest-risk dependency. Mitigated by doctor-first build order.
- **MPS op gaps in the layered VAE specifically**: unknown until tested. Tracer's fallback detection makes this visible immediately.
- **fp16 activation overflow on the DiT**: known issue with base Qwen-Image; DrawThings published a fix; we port it. If it doesn't transfer cleanly to the layered variant, may need stage-specific dtype handling.
