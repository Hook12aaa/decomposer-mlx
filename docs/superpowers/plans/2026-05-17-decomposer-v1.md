# Decomposer v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Mac-local Python web app that decomposes a flat marketing image into N RGBA layers using Qwen-Image-Layered, with first-class performance observability (X-ray tracing of every pipeline stage).

**Architecture:** CLI-first design with FastAPI as a thin presentation layer. Core is a five-phase inference pipeline (text-encoder → DiT → VAE) with sequential module residency to fit 48 GB unified memory. Every stage is instrumented via a bespoke `Tracer` that exports structured JSON and Perfetto-compatible waterfall traces.

**Tech Stack:** Python 3.12, uv, FastAPI, uvicorn, typer, rich, diffusers (HEAD), torch (MPS), gguf, Pillow, pytest, ruff.

**Spec:** See `docs/superpowers/specs/2026-05-17-decomposer-design.md`.

**Hardware:** Apple Silicon M3 Max, 48 GB unified memory. Many integration tests require MPS and are marked `@pytest.mark.mps_required` (skipped in CI).

---

## File Map

Before tasks: here is every file this plan creates and what it owns.

```
decomposer/
├── pyproject.toml                       # uv project, deps, tool config
├── ruff.toml                            # lint config
├── pytest.ini                           # markers (mps_required), asyncio mode
├── decomposer/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py                     # Shared dataclasses (StageRecord, Report, Job)
│   │   ├── probes.py                    # RSS + MPS alloc snapshots; CPU-fallback hook
│   │   ├── xray.py                      # Tracer (stage/step/annotate/report)
│   │   ├── perfetto.py                  # Perfetto JSON exporter
│   │   ├── residency.py                 # ResidencyManager — at-most-one MPS module
│   │   ├── backend.py                   # InferenceBackend Protocol + FakeBackend
│   │   ├── gguf_loader.py               # GgufLinear: Q8 dequant-on-the-fly module
│   │   ├── mps_backend.py               # MpsBackend.decompose(): the five phases
│   │   └── pipeline.py                  # Thin wrapper: orchestrates phases via tracer
│   ├── cli.py                           # typer entry: doctor / decompose / diff-traces
│   ├── web/
│   │   ├── __init__.py
│   │   ├── app.py                       # FastAPI routes + lifespan
│   │   ├── jobs.py                      # In-memory Job store + event broker
│   │   └── templates/index.html         # one page, vanilla HTML + SSE JS
│   └── test_assets/
│       ├── sample_ad_1.png              # bundled marketing samples
│       ├── sample_ad_2.png
│       └── tiny_smoke_test.png          # 256px image for `doctor`
└── tests/
    ├── conftest.py                      # pytest fixtures + mps_required marker
    ├── test_types.py
    ├── test_probes.py
    ├── test_xray.py
    ├── test_perfetto.py
    ├── test_residency.py
    ├── test_backend_fake.py
    ├── test_gguf_loader.py              # mps_required for real-dequant test
    ├── test_mps_backend.py              # mps_required for end-to-end
    ├── test_cli.py
    ├── test_web_jobs.py
    ├── test_web_app.py                  # uses FakeBackend
    └── test_diff_traces.py
```

---

## Task 1: Project skeleton & tooling

**Files:**
- Create: `pyproject.toml`, `ruff.toml`, `pytest.ini`, `.gitignore`
- Create: `decomposer/__init__.py`, `decomposer/core/__init__.py`, `decomposer/web/__init__.py`, `tests/conftest.py`

- [ ] **Step 1: Verify uv is installed**

Run: `uv --version`
Expected: `uv 0.5.x` or later. If missing, install with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "decomposer"
version = "0.1.0"
description = "Local Qwen-Image-Layered marketing-asset decomposition pipeline"
requires-python = ">=3.12"
dependencies = [
    "torch>=2.5.0",
    "diffusers @ git+https://github.com/huggingface/diffusers.git",
    "transformers>=4.51.3",
    "accelerate>=1.0.0",
    "gguf>=0.10.0",
    "safetensors>=0.4.5",
    "pillow>=10.0.0",
    "typer>=0.12.0",
    "rich>=13.0.0",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "jinja2>=3.1.0",
    "sse-starlette>=2.1.0",
    "psutil>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-mock>=3.14.0",
    "httpx>=0.27.0",
    "ruff>=0.7.0",
]

[project.scripts]
decomposer = "decomposer.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["decomposer"]
```

- [ ] **Step 3: Create `ruff.toml`**

```toml
line-length = 100
target-version = "py312"

[lint]
select = ["E", "F", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]
```

- [ ] **Step 4: Create `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
markers =
    mps_required: requires Apple Silicon with MPS available; skipped without it
testpaths = tests
```

- [ ] **Step 5: Create `.gitignore`**

```
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
runs/
dist/
*.egg-info/
*.safetensors
*.gguf
```

- [ ] **Step 6: Create empty package files**

Create `decomposer/__init__.py`:
```python
__version__ = "0.1.0"
```

Create `decomposer/core/__init__.py` and `decomposer/web/__init__.py` as empty files.

- [ ] **Step 7: Create `tests/conftest.py`**

```python
import pytest
import torch


def pytest_collection_modifyitems(config, items):
    if torch.backends.mps.is_available():
        return
    skip_mps = pytest.mark.skip(reason="requires Apple Silicon MPS")
    for item in items:
        if "mps_required" in item.keywords:
            item.add_marker(skip_mps)
```

- [ ] **Step 8: Install and verify**

Run: `uv sync --extra dev`
Then: `uv run pytest --collect-only`
Expected: `0 tests collected` (no tests yet, no errors)

Then: `uv run ruff check .`
Expected: clean

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml ruff.toml pytest.ini .gitignore decomposer/ tests/
git commit -m "Bootstrap decomposer project skeleton"
```

---

## Task 2: Core types

**Files:**
- Create: `decomposer/core/types.py`
- Test: `tests/test_types.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_types.py`:
```python
from decomposer.core.types import StageRecord, Report


def test_stage_record_required_fields():
    r = StageRecord(name="load_dit", wall_ms=42.0, gpu_ms=40.0,
                    rss_peak_mb=1024.0, mps_alloc_peak_mb=512.0,
                    mps_alloc_delta_mb=500.0, device_fallbacks=0,
                    extras={"quant": "q8_gguf"})
    assert r.name == "load_dit"
    assert r.extras["quant"] == "q8_gguf"


def test_report_aggregates_stages():
    r1 = StageRecord(name="a", wall_ms=10.0, gpu_ms=5.0, rss_peak_mb=100.0,
                     mps_alloc_peak_mb=50.0, mps_alloc_delta_mb=0.0,
                     device_fallbacks=0, extras={})
    r2 = StageRecord(name="b", wall_ms=20.0, gpu_ms=15.0, rss_peak_mb=120.0,
                     mps_alloc_peak_mb=80.0, mps_alloc_delta_mb=30.0,
                     device_fallbacks=2, extras={})
    rep = Report(stages=[r1, r2], total_wall_ms=30.0, run_id="run-1")
    assert rep.total_fallbacks() == 2
    assert rep.peak_mps_alloc_mb() == 80.0
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_types.py -v`
Expected: `ModuleNotFoundError: No module named 'decomposer.core.types'`

- [ ] **Step 3: Implement `decomposer/core/types.py`**

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageRecord:
    name: str
    wall_ms: float
    gpu_ms: float
    rss_peak_mb: float
    mps_alloc_peak_mb: float
    mps_alloc_delta_mb: float
    device_fallbacks: int
    extras: dict[str, Any] = field(default_factory=dict)
    steps: list["StageRecord"] = field(default_factory=list)


@dataclass
class Report:
    stages: list[StageRecord]
    total_wall_ms: float
    run_id: str

    def total_fallbacks(self) -> int:
        return sum(s.device_fallbacks for s in self.stages)

    def peak_mps_alloc_mb(self) -> float:
        return max((s.mps_alloc_peak_mb for s in self.stages), default=0.0)
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_types.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add decomposer/core/types.py tests/test_types.py
git commit -m "Add StageRecord and Report core types"
```

---

## Task 3: Memory & MPS probes

**Files:**
- Create: `decomposer/core/probes.py`
- Test: `tests/test_probes.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_probes.py`:
```python
from decomposer.core.probes import rss_mb, mps_alloc_mb, FallbackCounter


def test_rss_mb_returns_positive_number():
    assert rss_mb() > 0


def test_mps_alloc_mb_returns_number():
    assert mps_alloc_mb() >= 0


def test_fallback_counter_starts_at_zero():
    c = FallbackCounter()
    assert c.count == 0


def test_fallback_counter_increments_on_warning():
    c = FallbackCounter()
    c.note("aten::some_op fell back to CPU")
    c.note("aten::another fell back to CPU")
    assert c.count == 2


def test_fallback_counter_ignores_unrelated_warnings():
    c = FallbackCounter()
    c.note("UserWarning: something else")
    assert c.count == 0
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_probes.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/core/probes.py`**

```python
import os
import psutil
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def mps_alloc_mb() -> float:
    if not torch.backends.mps.is_available():
        return 0.0
    return torch.mps.driver_allocated_memory() / (1024 * 1024)


class FallbackCounter:
    def __init__(self) -> None:
        self.count = 0

    def note(self, message: str) -> None:
        if "fell back to CPU" in message or "MPS: " in message and "fallback" in message:
            self.count += 1
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_probes.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add decomposer/core/probes.py tests/test_probes.py
git commit -m "Add memory and MPS allocation probes plus fallback counter"
```

---

## Task 4: Tracer — stage & step API

**Files:**
- Create: `decomposer/core/xray.py`
- Test: `tests/test_xray.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_xray.py`:
```python
import time
from decomposer.core.xray import Tracer


def test_tracer_records_a_stage():
    t = Tracer(run_id="r1")
    with t.stage("load_dit", quant="q8"):
        time.sleep(0.01)
    report = t.report()
    assert len(report.stages) == 1
    s = report.stages[0]
    assert s.name == "load_dit"
    assert s.wall_ms >= 10.0
    assert s.extras["quant"] == "q8"


def test_tracer_records_steps_within_a_stage():
    t = Tracer(run_id="r2")
    with t.stage("denoise_loop", steps=3):
        for i in range(3):
            with t.step("denoise_step", i=i):
                time.sleep(0.005)
    report = t.report()
    assert len(report.stages) == 1
    assert len(report.stages[0].steps) == 3
    assert all(s.name == "denoise_step" for s in report.stages[0].steps)


def test_tracer_annotate_adds_to_current_stage():
    t = Tracer(run_id="r3")
    with t.stage("encode_prompt"):
        t.annotate(token_count=128)
    report = t.report()
    assert report.stages[0].extras["token_count"] == 128


def test_tracer_increments_fallback_count_on_warning():
    import warnings
    t = Tracer(run_id="r-fb")
    with t.stage("denoise_step"):
        warnings.warn("aten::some_op fell back to CPU on MPS", UserWarning)
    report = t.report()
    assert report.stages[0].device_fallbacks == 1


def test_tracer_total_wall_sums_stage_walls():
    t = Tracer(run_id="r4")
    with t.stage("a"):
        time.sleep(0.01)
    with t.stage("b"):
        time.sleep(0.01)
    report = t.report()
    assert report.total_wall_ms >= 20.0
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_xray.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/core/xray.py`**

```python
import time
from contextlib import contextmanager
from typing import Any, Iterator

from decomposer.core.probes import FallbackCounter, mps_alloc_mb, rss_mb
from decomposer.core.types import Report, StageRecord


class Tracer:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._stages: list[StageRecord] = []
        self._stack: list[StageRecord] = []
        self._listeners: list = []
        self._install_fallback_hook()

    def _install_fallback_hook(self) -> None:
        import warnings

        tracer_ref = self

        def showwarning(message, category, filename, lineno, file=None, line=None):
            text = str(message)
            if "fell back to CPU" in text or "MPS:" in text and "fallback" in text:
                if tracer_ref._stack:
                    tracer_ref._stack[-1].device_fallbacks += 1
            warnings._original_showwarning(message, category, filename, lineno, file, line)

        if not hasattr(warnings, "_original_showwarning"):
            warnings._original_showwarning = warnings.showwarning
            warnings.showwarning = showwarning

    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def _emit(self, event: str, **payload: Any) -> None:
        for fn in self._listeners:
            fn(event, payload)

    @contextmanager
    def stage(self, name: str, **extras: Any) -> Iterator[None]:
        rec = self._begin(name, extras)
        try:
            yield
        finally:
            self._end(rec)
            if not self._stack:
                self._stages.append(rec)

    @contextmanager
    def step(self, name: str, **extras: Any) -> Iterator[None]:
        if not self._stack:
            raise RuntimeError("step() requires an active stage")
        rec = self._begin(name, extras, as_step=True)
        try:
            yield
        finally:
            self._end(rec)

    def annotate(self, **extras: Any) -> None:
        if not self._stack:
            raise RuntimeError("annotate() requires an active stage")
        self._stack[-1].extras.update(extras)

    def _begin(self, name: str, extras: dict[str, Any], *, as_step: bool = False) -> StageRecord:
        parent = self._stack[-1] if self._stack else None
        rec = StageRecord(
            name=name, wall_ms=0.0, gpu_ms=0.0,
            rss_peak_mb=rss_mb(), mps_alloc_peak_mb=mps_alloc_mb(),
            mps_alloc_delta_mb=0.0, device_fallbacks=0, extras=dict(extras),
        )
        rec.extras["_start_ts"] = time.perf_counter()
        rec.extras["_start_mps"] = mps_alloc_mb()
        if as_step and parent is not None:
            parent.steps.append(rec)
        self._stack.append(rec)
        self._emit("stage_started", name=name, extras=extras)
        return rec

    def _end(self, rec: StageRecord) -> None:
        start = rec.extras.pop("_start_ts")
        start_mps = rec.extras.pop("_start_mps")
        rec.wall_ms = (time.perf_counter() - start) * 1000.0
        end_mps = mps_alloc_mb()
        rec.mps_alloc_peak_mb = max(rec.mps_alloc_peak_mb, end_mps)
        rec.mps_alloc_delta_mb = end_mps - start_mps
        rec.rss_peak_mb = max(rec.rss_peak_mb, rss_mb())
        self._stack.pop()
        self._emit("stage_ended", name=rec.name, wall_ms=rec.wall_ms)

    def report(self) -> Report:
        total = sum(s.wall_ms for s in self._stages)
        return Report(stages=list(self._stages), total_wall_ms=total, run_id=self.run_id)
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_xray.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add decomposer/core/xray.py tests/test_xray.py
git commit -m "Add Tracer with nested stages, steps, and annotation"
```

---

## Task 5: Tracer JSON & Perfetto exports

**Files:**
- Create: `decomposer/core/perfetto.py`
- Modify: `decomposer/core/xray.py` (add `to_json` helper)
- Test: `tests/test_perfetto.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_perfetto.py`:
```python
import json
from decomposer.core.xray import Tracer
from decomposer.core.perfetto import to_perfetto, report_to_json


def make_report():
    t = Tracer(run_id="run-test")
    with t.stage("load_dit"):
        with t.stage("inner"):
            pass
    with t.stage("denoise_loop", steps=2):
        with t.step("denoise_step", i=0):
            pass
        with t.step("denoise_step", i=1):
            pass
    return t.report()


def test_report_to_json_serializes_round_trip():
    rep = make_report()
    raw = report_to_json(rep)
    parsed = json.loads(raw)
    assert parsed["run_id"] == "run-test"
    assert len(parsed["stages"]) == 2
    assert parsed["stages"][1]["name"] == "denoise_loop"
    assert len(parsed["stages"][1]["steps"]) == 2


def test_perfetto_emits_trace_events_array():
    rep = make_report()
    trace = to_perfetto(rep)
    assert "traceEvents" in trace
    events = trace["traceEvents"]
    names = [e["name"] for e in events if e["ph"] == "X"]
    assert "load_dit" in names
    assert "denoise_step" in names
    for e in events:
        if e["ph"] == "X":
            assert "ts" in e and "dur" in e and "pid" in e
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_perfetto.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/core/perfetto.py`**

```python
import json
from dataclasses import asdict

from decomposer.core.types import Report, StageRecord


def report_to_json(report: Report) -> str:
    return json.dumps(asdict(report), indent=2, default=str)


def to_perfetto(report: Report) -> dict:
    events: list[dict] = []
    cursor_us = 0

    def emit(rec: StageRecord, depth: int) -> None:
        nonlocal cursor_us
        start = cursor_us
        events.append({
            "name": rec.name,
            "cat": "stage" if depth == 0 else "step",
            "ph": "X",
            "ts": start,
            "dur": int(rec.wall_ms * 1000),
            "pid": 1,
            "tid": depth,
            "args": {
                "gpu_ms": rec.gpu_ms,
                "mps_alloc_peak_mb": rec.mps_alloc_peak_mb,
                "rss_peak_mb": rec.rss_peak_mb,
                "device_fallbacks": rec.device_fallbacks,
                **rec.extras,
            },
        })
        inner_cursor = start
        for child in rec.steps:
            cursor_us = inner_cursor
            emit(child, depth + 1)
            inner_cursor = cursor_us
        cursor_us = start + int(rec.wall_ms * 1000)

    for s in report.stages:
        emit(s, 0)

    return {"traceEvents": events, "displayTimeUnit": "ms"}
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_perfetto.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add decomposer/core/perfetto.py tests/test_perfetto.py
git commit -m "Add JSON and Perfetto trace exporters"
```

---

## Task 6: ResidencyManager

**Files:**
- Create: `decomposer/core/residency.py`
- Test: `tests/test_residency.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_residency.py`:
```python
import pytest
from decomposer.core.residency import ResidencyManager


class FakeModule:
    def __init__(self, name: str) -> None:
        self.name = name
        self.on_device = False

    def to(self, device: str) -> "FakeModule":
        self.on_device = (device != "cpu")
        return self


def test_residency_loads_module_to_mps():
    rm = ResidencyManager(device="cpu")
    loaded = rm.load("text", lambda: FakeModule("text"))
    assert rm.current_name == "text"
    assert loaded.name == "text"


def test_residency_frees_previous_before_loading_next():
    rm = ResidencyManager(device="cpu")
    loaded_order: list[str] = []
    rm.load("text", lambda: (loaded_order.append("load-text"), FakeModule("text"))[1])
    rm.load("dit", lambda: (loaded_order.append("load-dit"), FakeModule("dit"))[1])
    assert loaded_order == ["load-text", "load-dit"]
    assert rm.current_name == "dit"


def test_residency_free_is_idempotent():
    rm = ResidencyManager(device="cpu")
    rm.load("text", lambda: FakeModule("text"))
    rm.free()
    rm.free()
    assert rm.current_name is None


def test_residency_rejects_unknown_module_name():
    rm = ResidencyManager(device="cpu")
    with pytest.raises(ValueError, match="unknown module"):
        rm.load("not_a_module", lambda: FakeModule("x"))
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_residency.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/core/residency.py`**

```python
import gc
from typing import Callable, Literal

import torch


ALLOWED = {"text", "dit", "vae"}


class ResidencyManager:
    def __init__(self, device: str = "mps") -> None:
        self.device = device
        self._current: object | None = None
        self.current_name: str | None = None

    def load(self, name: Literal["text", "dit", "vae"], factory: Callable[[], object]) -> object:
        if name not in ALLOWED:
            raise ValueError(f"unknown module {name!r}; allowed={sorted(ALLOWED)}")
        if self._current is not None:
            self.free()
        module = factory()
        if hasattr(module, "to"):
            module = module.to(self.device)
        self._current = module
        self.current_name = name
        return module

    def free(self) -> None:
        if self._current is None:
            return
        self._current = None
        self.current_name = None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_residency.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add decomposer/core/residency.py tests/test_residency.py
git commit -m "Add ResidencyManager enforcing at-most-one MPS module"
```

---

## Task 7: InferenceBackend Protocol + FakeBackend

**Files:**
- Create: `decomposer/core/backend.py`
- Test: `tests/test_backend_fake.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_backend_fake.py`:
```python
from PIL import Image
from decomposer.core.backend import FakeBackend
from decomposer.core.xray import Tracer


def test_fake_backend_returns_n_rgba_layers():
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    backend = FakeBackend(latency_ms=10)
    layers = backend.decompose(img, layers=6, resolution=256, steps=4)
    assert len(layers) == 6
    for layer in layers:
        assert layer.mode == "RGBA"
        assert layer.size == (256, 256)


def test_fake_backend_records_to_tracer():
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    backend = FakeBackend(latency_ms=5)
    t = Tracer(run_id="run-x")
    backend.decompose(img, layers=4, resolution=128, steps=4, tracer=t)
    rep = t.report()
    names = [s.name for s in rep.stages]
    for expected in ["load_text_encoder", "encode_prompt", "load_dit",
                     "denoise_loop", "load_vae", "decode_layers"]:
        assert expected in names
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_backend_fake.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/core/backend.py`**

```python
import time
from typing import Protocol

from PIL import Image

from decomposer.core.xray import Tracer


class InferenceBackend(Protocol):
    def decompose(
        self,
        image: Image.Image,
        layers: int,
        resolution: int = 640,
        steps: int = 8,
        seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> list[Image.Image]: ...


class FakeBackend:
    def __init__(self, latency_ms: int = 100) -> None:
        self._latency_s = latency_ms / 1000.0

    def decompose(
        self,
        image: Image.Image,
        layers: int,
        resolution: int = 640,
        steps: int = 8,
        seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> list[Image.Image]:
        t = tracer or Tracer(run_id="fake")
        with t.stage("load_text_encoder"):
            time.sleep(self._latency_s)
        with t.stage("encode_prompt"):
            time.sleep(self._latency_s)
        with t.stage("free_text_encoder"):
            pass
        with t.stage("load_dit"):
            time.sleep(self._latency_s)
        with t.stage("denoise_loop", steps=steps):
            for i in range(steps):
                with t.step("denoise_step", i=i):
                    time.sleep(self._latency_s / steps)
        with t.stage("free_dit"):
            pass
        with t.stage("load_vae"):
            time.sleep(self._latency_s)
        with t.stage("decode_layers", n=layers):
            time.sleep(self._latency_s)
        with t.stage("free_vae"):
            pass
        return [Image.new("RGBA", (resolution, resolution), (i * 30, 0, 0, 255))
                for i in range(layers)]
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_backend_fake.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add decomposer/core/backend.py tests/test_backend_fake.py
git commit -m "Add InferenceBackend protocol and FakeBackend for testing"
```

---

## Task 8: GgufLinear — Q8 dequantization module

> **Risk gate:** This task implements the highest-risk piece of v1. The unit test verifies dequant math; the `mps_required` test verifies it actually runs on Apple Silicon. If the MPS test fails, **stop and revisit** with real evidence rather than proceeding.

**Files:**
- Create: `decomposer/core/gguf_loader.py`
- Test: `tests/test_gguf_loader.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_gguf_loader.py`:
```python
import numpy as np
import pytest
import torch

from decomposer.core.gguf_loader import GgufLinear, dequantize_q8_0


def test_dequantize_q8_0_recovers_approx_original():
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((32, 32)).astype(np.float32) * 0.1
    quantized = _pack_q8_0(weight)
    dequantized = dequantize_q8_0(quantized, shape=(32, 32))
    err = np.abs(dequantized - weight).max()
    assert err < 0.01, f"dequant error too high: {err}"


def test_gguf_linear_forward_matches_fp_reference():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((16, 8)).astype(np.float32) * 0.1
    bias = rng.standard_normal(16).astype(np.float32) * 0.1
    x = torch.tensor(rng.standard_normal((4, 8)).astype(np.float32))

    quantized = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=quantized, shape=(16, 8),
                      bias=torch.tensor(bias), dtype=torch.float32)

    ref = x @ torch.tensor(weight).T + torch.tensor(bias)
    out = layer(x)
    err = (out - ref).abs().max().item()
    assert err < 0.05, f"forward error too high: {err}"


@pytest.mark.mps_required
def test_gguf_linear_runs_on_mps():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((64, 32)).astype(np.float32) * 0.1
    quantized = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=quantized, shape=(64, 32),
                      bias=None, dtype=torch.float16).to("mps")
    x = torch.randn(2, 32, device="mps", dtype=torch.float16)
    out = layer(x)
    assert out.device.type == "mps"
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def _pack_q8_0(weight: np.ndarray) -> bytes:
    """Pack a 2-D float32 weight into Q8_0 GGUF block format.
    Block layout: 2 bytes fp16 scale, 32 bytes int8 quants per 32-element block.
    """
    flat = weight.reshape(-1)
    assert flat.size % 32 == 0
    out = bytearray()
    for i in range(0, flat.size, 32):
        block = flat[i:i + 32]
        scale = float(np.abs(block).max() / 127.0) if np.any(block) else 1e-8
        quant = np.clip(np.round(block / scale), -127, 127).astype(np.int8)
        out += np.float16(scale).tobytes()
        out += quant.tobytes()
    return bytes(out)
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_gguf_loader.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/core/gguf_loader.py`**

```python
import numpy as np
import torch
import torch.nn as nn


Q8_BLOCK_SIZE = 32
Q8_BLOCK_BYTES = 2 + Q8_BLOCK_SIZE


def dequantize_q8_0(packed: bytes, shape: tuple[int, ...]) -> np.ndarray:
    n_elements = int(np.prod(shape))
    assert n_elements % Q8_BLOCK_SIZE == 0
    n_blocks = n_elements // Q8_BLOCK_SIZE
    assert len(packed) == n_blocks * Q8_BLOCK_BYTES

    out = np.empty(n_elements, dtype=np.float32)
    arr = np.frombuffer(packed, dtype=np.uint8)
    for b in range(n_blocks):
        offset = b * Q8_BLOCK_BYTES
        scale = np.frombuffer(arr[offset:offset + 2].tobytes(), dtype=np.float16)[0]
        quants = np.frombuffer(arr[offset + 2:offset + Q8_BLOCK_BYTES].tobytes(),
                               dtype=np.int8).astype(np.float32)
        out[b * Q8_BLOCK_SIZE:(b + 1) * Q8_BLOCK_SIZE] = quants * float(scale)
    return out.reshape(shape)


class GgufLinear(nn.Module):
    def __init__(self, *, quantized_weight: bytes, shape: tuple[int, int],
                 bias: torch.Tensor | None, dtype: torch.dtype) -> None:
        super().__init__()
        self._packed = quantized_weight
        self._shape = shape
        self._dtype = dtype
        weight = dequantize_q8_0(quantized_weight, shape)
        self.register_buffer("weight", torch.from_numpy(weight).to(dtype))
        self.bias = nn.Parameter(bias.to(dtype)) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
```

- [ ] **Step 4: Run unit tests**

Run: `uv run pytest tests/test_gguf_loader.py -v -m "not mps_required"`
Expected: 2 passed

- [ ] **Step 5: Run MPS smoke test**

Run: `uv run pytest tests/test_gguf_loader.py::test_gguf_linear_runs_on_mps -v`
Expected (on M3 Max): 1 passed
Expected (no MPS): 1 skipped

**If this test fails with a real MPS error, STOP. Investigate before proceeding.**

- [ ] **Step 6: Commit**

```bash
git add decomposer/core/gguf_loader.py tests/test_gguf_loader.py
git commit -m "Add GgufLinear: Q8_0 dequant-on-forward module"
```

---

## Task 9: MpsBackend — text encoder phase

**Files:**
- Create: `decomposer/core/mps_backend.py`
- Test: `tests/test_mps_backend.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_mps_backend.py`:
```python
import pytest
from PIL import Image

from decomposer.core.mps_backend import MpsBackend
from decomposer.core.xray import Tracer


@pytest.mark.mps_required
def test_mps_backend_text_encoder_phase_runs():
    backend = MpsBackend()
    t = Tracer(run_id="r-te")
    cond = backend._encode_prompt(
        image=Image.new("RGB", (128, 128), (0, 0, 0)),
        prompt="a marketing advertisement",
        tracer=t,
    )
    assert cond is not None
    names = [s.name for s in t.report().stages]
    assert "load_text_encoder" in names
    assert "encode_prompt" in names
    assert "free_text_encoder" in names
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_mps_backend.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement skeleton of `decomposer/core/mps_backend.py`**

```python
import asyncio
import gc

import torch
from PIL import Image

from decomposer.core.residency import ResidencyManager
from decomposer.core.xray import Tracer


HF_REPO = "Qwen/Qwen-Image-Layered"
TEXT_ENCODER_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"


class MpsBackend:
    def __init__(self, device: str = "mps", dtype: torch.dtype = torch.float16,
                 lightning_lora_path: str | None = None) -> None:
        self.device = device
        self.dtype = dtype
        self.lightning_lora_path = lightning_lora_path
        self.residency = ResidencyManager(device=device)
        self._lock = asyncio.Lock()

    def _encode_prompt(self, image: Image.Image, prompt: str, *, tracer: Tracer):
        from transformers import AutoModel, AutoProcessor

        with tracer.stage("load_text_encoder"):
            te = self.residency.load("text", lambda: AutoModel.from_pretrained(
                TEXT_ENCODER_REPO, torch_dtype=self.dtype))
            processor = AutoProcessor.from_pretrained(TEXT_ENCODER_REPO)

        with tracer.stage("encode_prompt", image_size=image.size):
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = te(**inputs)
            cond = outputs.last_hidden_state.detach().to("cpu")
            tracer.annotate(token_count=int(inputs["input_ids"].shape[-1]))

        with tracer.stage("free_text_encoder"):
            del te, processor, inputs, outputs
            self.residency.free()
            gc.collect()

        return cond
```

- [ ] **Step 4: Run MPS test (downloads ~14 GB of weights the first time — be patient)**

Run: `uv run pytest tests/test_mps_backend.py::test_mps_backend_text_encoder_phase_runs -v`
Expected: 1 passed (or 1 skipped on non-MPS)

If this fails: read the traceback, surface to user, do not paper over.

- [ ] **Step 5: Commit**

```bash
git add decomposer/core/mps_backend.py tests/test_mps_backend.py
git commit -m "Add MpsBackend skeleton with text-encoder phase"
```

---

## Task 10: MpsBackend — DiT + denoise + VAE phases

**Files:**
- Modify: `decomposer/core/mps_backend.py`
- Modify: `tests/test_mps_backend.py`

- [ ] **Step 1: Add failing integration test**

Append to `tests/test_mps_backend.py`:
```python
@pytest.mark.mps_required
def test_mps_backend_decompose_returns_layers():
    backend = MpsBackend()
    img = Image.new("RGB", (256, 256), (200, 50, 50))
    t = Tracer(run_id="r-full")
    layers = backend.decompose(img, layers=4, resolution=256, steps=4, tracer=t)
    assert len(layers) == 4
    assert all(layer.mode == "RGBA" for layer in layers)
    assert all(layer.size == (256, 256) for layer in layers)
    rep = t.report()
    expected = ["load_text_encoder", "encode_prompt", "free_text_encoder",
                "load_dit", "denoise_loop", "free_dit",
                "load_vae", "decode_layers", "free_vae"]
    seen = [s.name for s in rep.stages]
    for name in expected:
        assert name in seen, f"missing stage: {name}"
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_mps_backend.py::test_mps_backend_decompose_returns_layers -v`
Expected: AttributeError (no `decompose` yet)

- [ ] **Step 3: Implement `decompose` and supporting methods**

Append to `decomposer/core/mps_backend.py`:
```python
    def decompose(
        self,
        image: Image.Image,
        layers: int,
        resolution: int = 640,
        steps: int = 8,
        seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> list[Image.Image]:
        t = tracer or Tracer(run_id="adhoc")
        cond = self._encode_prompt(image, prompt="marketing asset", tracer=t)
        latent = self._denoise(cond, image=image, layers=layers,
                               resolution=resolution, steps=steps, seed=seed, tracer=t)
        return self._decode(latent, layers=layers, resolution=resolution, tracer=t)

    def _denoise(self, cond, *, image, layers, resolution, steps, seed, tracer: Tracer):
        from diffusers import QwenImageLayeredPipeline

        with tracer.stage("load_dit", quant="q8_gguf", dtype=str(self.dtype)):
            pipe = self.residency.load("dit", lambda: QwenImageLayeredPipeline.from_pretrained(
                HF_REPO, torch_dtype=self.dtype))
            pipe.text_encoder = None
            if self.lightning_lora_path is not None:
                pipe.load_lora_weights(self.lightning_lora_path)
                tracer.annotate(lightning_lora=True)

        with tracer.stage("denoise_loop", steps=steps) as _:
            generator = torch.Generator(device="cpu")
            if seed is not None:
                generator.manual_seed(seed)

            def step_cb(pipeline, step_index, timestep, callback_kwargs):
                if not hasattr(step_cb, "_ctx"):
                    step_cb._ctx = None
                if step_cb._ctx is not None:
                    step_cb._ctx.__exit__(None, None, None)
                ctx = tracer.step("denoise_step", i=step_index, t=int(timestep))
                ctx.__enter__()
                step_cb._ctx = ctx
                return callback_kwargs

            latents = pipe(
                image=image, layers=layers, resolution=resolution,
                num_inference_steps=steps, true_cfg_scale=1.0,
                generator=generator, output_type="latent",
                callback_on_step_end=step_cb,
            ).images
            if getattr(step_cb, "_ctx", None) is not None:
                step_cb._ctx.__exit__(None, None, None)
                step_cb._ctx = None

        with tracer.stage("free_dit"):
            del pipe
            self.residency.free()
            gc.collect()

        return latents

    def _decode(self, latents, *, layers, resolution, tracer: Tracer) -> list[Image.Image]:
        from diffusers import AutoencoderKLQwenImage

        with tracer.stage("load_vae"):
            vae = self.residency.load("vae", lambda: AutoencoderKLQwenImage.from_pretrained(
                HF_REPO, subfolder="vae", torch_dtype=self.dtype))

        with tracer.stage("decode_layers", n=layers):
            with torch.no_grad():
                decoded = vae.decode(latents.to(self.device)).sample

        with tracer.stage("free_vae"):
            del vae
            self.residency.free()
            gc.collect()

        decoded = decoded.clamp(0, 1).to(torch.float32).cpu().numpy()
        return [_tensor_to_rgba(decoded[i]) for i in range(layers)]


def _tensor_to_rgba(arr) -> Image.Image:
    import numpy as np
    if arr.shape[0] == 4:
        arr = arr.transpose(1, 2, 0)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")
```

**Note on Lightning LoRA:** The `lightning_lora_path` arg lets you load a Lightning LoRA when one is published for Qwen-Image-Layered (none confirmed at the time of this plan). Without it, 8 steps will still run but with degraded quality vs. the 50-step baseline — this is an acceptable v1 tradeoff per the spec.

- [ ] **Step 4: Run MPS integration test (slow — many minutes first time)**

Run: `uv run pytest tests/test_mps_backend.py::test_mps_backend_decompose_returns_layers -v -s`
Expected: 1 passed (or skipped on non-MPS)

If this fails, the exact stage that failed is in the tracer report — surface that diagnosis.

- [ ] **Step 5: Commit**

```bash
git add decomposer/core/mps_backend.py tests/test_mps_backend.py
git commit -m "Implement MpsBackend.decompose end-to-end on MPS"
```

---

## Task 11: CLI — `decomposer doctor`

**Files:**
- Create: `decomposer/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_cli.py`:
```python
from typer.testing import CliRunner
from decomposer.cli import app

runner = CliRunner()


def test_doctor_fake_backend_passes():
    result = runner.invoke(app, ["doctor", "--fake"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output or "OK" in result.output


def test_decompose_with_fake_writes_layers(tmp_path):
    from PIL import Image
    src = tmp_path / "in.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(src)
    out = tmp_path / "out"
    result = runner.invoke(app, ["decompose", str(src), "--layers", "3",
                                  "--out", str(out), "--fake"])
    assert result.exit_code == 0, result.output
    pngs = list(out.glob("layer_*.png"))
    assert len(pngs) == 3
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_cli.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/cli.py`**

```python
import json
import time
from pathlib import Path
from typing import Annotated, Optional

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

from decomposer.core.backend import FakeBackend, InferenceBackend
from decomposer.core.perfetto import report_to_json, to_perfetto
from decomposer.core.xray import Tracer

app = typer.Typer(no_args_is_help=True)
console = Console()


def _backend(fake: bool) -> InferenceBackend:
    if fake:
        return FakeBackend(latency_ms=50)
    from decomposer.core.mps_backend import MpsBackend
    return MpsBackend()


@app.command()
def doctor(
    fake: Annotated[bool, typer.Option(help="Use FakeBackend (no model load)")] = False,
) -> None:
    """Smoke-test the install: imports, MPS, Q8 GGUF, mini decomposition."""
    import torch

    console.print(f"[bold]decomposer doctor[/bold] — fake={fake}")
    console.print(f"  torch: {torch.__version__}")
    console.print(f"  MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        console.print(f"  MPS recommended max mem: {torch.mps.recommended_max_memory() / 1e9:.1f} GB")

    img_path = Path(__file__).parent / "test_assets" / "tiny_smoke_test.png"
    if not img_path.exists():
        img = Image.new("RGB", (256, 256), (180, 80, 80))
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_path)
    img = Image.open(img_path).convert("RGB")

    backend = _backend(fake)
    t = Tracer(run_id=f"doctor-{int(time.time())}")
    try:
        layers = backend.decompose(img, layers=3, resolution=256, steps=4, tracer=t)
    except Exception as e:
        console.print(f"[red]FAIL[/red]: {e}")
        raise typer.Exit(1)

    _print_report(t.report())
    if len(layers) == 3:
        console.print("[green]PASS[/green] — 3 layers returned")
    else:
        console.print(f"[red]FAIL[/red] — expected 3 layers, got {len(layers)}")
        raise typer.Exit(1)


@app.command()
def decompose(
    image: Annotated[Path, typer.Argument(exists=True, readable=True)],
    layers: int = 6,
    resolution: int = 640,
    steps: int = 8,
    seed: Optional[int] = None,
    out: Path = Path("./out"),
    trace: bool = True,
    fake: bool = False,
) -> None:
    """Decompose IMAGE into N RGBA layers."""
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(image).convert("RGB")
    backend = _backend(fake)
    t = Tracer(run_id=f"run-{int(time.time())}")

    console.print(f"[bold]Decomposing[/bold] {image} → {out} (layers={layers}, res={resolution}, steps={steps})")
    result = backend.decompose(img, layers=layers, resolution=resolution,
                                steps=steps, seed=seed, tracer=t)

    for i, layer in enumerate(result):
        layer.save(out / f"layer_{i}.png")
    rep = t.report()
    if trace:
        (out / "trace.json").write_text(report_to_json(rep))
        (out / "trace.perfetto.json").write_text(json.dumps(to_perfetto(rep), indent=2))

    _print_report(rep)
    console.print(f"[green]Wrote {len(result)} layers to {out}[/green]")


@app.command("diff-traces")
def diff_traces(run_a: Path, run_b: Path) -> None:
    """Compare two trace.json reports side by side."""
    a = json.loads(Path(run_a).read_text())
    b = json.loads(Path(run_b).read_text())
    by_name_a = {s["name"]: s for s in a["stages"]}
    by_name_b = {s["name"]: s for s in b["stages"]}
    all_names = sorted(set(by_name_a) | set(by_name_b))

    table = Table(title=f"{run_a.name} vs {run_b.name}")
    table.add_column("Stage")
    table.add_column("A wall_ms", justify="right")
    table.add_column("B wall_ms", justify="right")
    table.add_column("Δ wall_ms", justify="right")
    table.add_column("A peak MB", justify="right")
    table.add_column("B peak MB", justify="right")

    for name in all_names:
        sa = by_name_a.get(name, {})
        sb = by_name_b.get(name, {})
        wa = float(sa.get("wall_ms", 0))
        wb = float(sb.get("wall_ms", 0))
        delta = wb - wa
        color = "red" if delta > 0 else "green"
        table.add_row(
            name, f"{wa:.1f}", f"{wb:.1f}",
            f"[{color}]{delta:+.1f}[/{color}]",
            f"{sa.get('mps_alloc_peak_mb', 0):.0f}",
            f"{sb.get('mps_alloc_peak_mb', 0):.0f}",
        )
    console.print(table)


def _print_report(rep) -> None:
    table = Table(title=f"Run {rep.run_id} — total {rep.total_wall_ms:.0f} ms")
    table.add_column("Stage")
    table.add_column("Wall ms", justify="right")
    table.add_column("Peak MPS MB", justify="right")
    table.add_column("Fallbacks", justify="right")
    for s in rep.stages:
        table.add_row(s.name, f"{s.wall_ms:.1f}",
                      f"{s.mps_alloc_peak_mb:.0f}", str(s.device_fallbacks))
    console.print(table)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Run doctor manually with fake backend**

Run: `uv run decomposer doctor --fake`
Expected: exit 0, "PASS — 3 layers returned"

- [ ] **Step 6: Commit**

```bash
git add decomposer/cli.py tests/test_cli.py
git commit -m "Add CLI: doctor, decompose, diff-traces"
```

---

## Task 12: FastAPI — Job store & event broker

**Files:**
- Create: `decomposer/web/jobs.py`
- Test: `tests/test_web_jobs.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_web_jobs.py`:
```python
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
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_web_jobs.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/web/jobs.py`**

```python
import asyncio
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Literal

JobStatus = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    status: JobStatus = "queued"
    stage: str | None = None
    output_dir: Path | None = None
    error: str | None = None


@dataclass
class JobStore:
    _jobs: dict[str, Job] = field(default_factory=dict)
    _queues: dict[str, list[asyncio.Queue]] = field(default_factory=dict)

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:12])
        self._jobs[job.id] = job
        self._queues[job.id] = []
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def publish(self, job_id: str, event: str, payload: dict) -> None:
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
```

- [ ] **Step 4: Run test to confirm pass**

Run: `uv run pytest tests/test_web_jobs.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add decomposer/web/jobs.py tests/test_web_jobs.py
git commit -m "Add JobStore with pub/sub for SSE delivery"
```

---

## Task 13: FastAPI — app, lifespan, routes

**Files:**
- Create: `decomposer/web/app.py`
- Create: `decomposer/web/templates/index.html`
- Test: `tests/test_web_app.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_web_app.py`:
```python
import pytest
from httpx import ASGITransport, AsyncClient
from decomposer.web.app import create_app
from decomposer.core.backend import FakeBackend


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
        r = await client.post("/jobs", json={"sample": "tiny_smoke_test.png",
                                              "layers": 3, "resolution": 128, "steps": 4})
        assert r.status_code == 200
        body = r.json()
        assert "job_id" in body and "stream" in body


@pytest.mark.asyncio
async def test_zip_route_returns_404_before_done(client_factory):
    async with client_factory() as client:
        r = await client.get("/jobs/nonexistent/zip")
        assert r.status_code == 404
```

- [ ] **Step 2: Create `decomposer/web/templates/index.html`**

```html
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Decomposer</title>
  <style>
    body { font-family: -apple-system, sans-serif; max-width: 720px; margin: 2em auto; padding: 0 1em; }
    button { padding: 0.5em 1em; font-size: 1em; }
    #log { font-family: monospace; background: #f4f4f4; padding: 1em; min-height: 200px; white-space: pre-wrap; }
    .stage { color: #06c; }
    .done { color: #060; font-weight: bold; }
    .err { color: #c00; }
  </style>
</head>
<body>
  <h1>Decomposer</h1>
  <label>Sample image:
    <select id="sample">
      <option value="sample_ad_1.png">sample_ad_1</option>
      <option value="sample_ad_2.png">sample_ad_2</option>
      <option value="tiny_smoke_test.png">tiny_smoke_test (256px)</option>
    </select>
  </label>
  <label>Layers: <input id="layers" type="number" value="6" min="2" max="12"></label>
  <label>Resolution: <input id="res" type="number" value="640"></label>
  <label>Steps: <input id="steps" type="number" value="8"></label>
  <p><button id="go">Decompose</button></p>
  <div id="log"></div>
  <p id="dl"></p>
  <script>
    const log = document.getElementById('log');
    const dl = document.getElementById('dl');
    function append(line, cls) {
      const span = document.createElement('div');
      if (cls) span.className = cls;
      span.textContent = line;
      log.appendChild(span);
    }
    document.getElementById('go').onclick = async () => {
      log.textContent = '';
      dl.textContent = '';
      const body = {
        sample: document.getElementById('sample').value,
        layers: +document.getElementById('layers').value,
        resolution: +document.getElementById('res').value,
        steps: +document.getElementById('steps').value,
      };
      const r = await fetch('/jobs', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
      const j = await r.json();
      append('Job ' + j.job_id + ' started');
      const es = new EventSource(j.stream);
      es.addEventListener('stage_started', e => append('▶ ' + JSON.parse(e.data).name, 'stage'));
      es.addEventListener('stage_ended', e => {
        const p = JSON.parse(e.data);
        append('✓ ' + p.name + ' (' + p.wall_ms.toFixed(0) + ' ms)');
      });
      es.addEventListener('done', () => {
        append('Done.', 'done');
        dl.innerHTML = '<a href="/jobs/' + j.job_id + '/zip">Download ZIP</a>';
        es.close();
      });
      es.addEventListener('error', e => { append('Error: ' + e.data, 'err'); es.close(); });
    };
  </script>
</body>
</html>
```

- [ ] **Step 3: Implement `decomposer/web/app.py`**

```python
import asyncio
import io
import json
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from decomposer.core.backend import InferenceBackend
from decomposer.core.perfetto import report_to_json, to_perfetto
from decomposer.core.xray import Tracer
from decomposer.web.jobs import JobStore

ASSETS = Path(__file__).resolve().parent.parent / "test_assets"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


class JobRequest(BaseModel):
    sample: str
    layers: int = 6
    resolution: int = 640
    steps: int = 8


def create_app(backend: InferenceBackend | None = None) -> FastAPI:
    store = JobStore()
    inference_lock = asyncio.Lock()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        nonlocal backend
        if backend is None:
            from decomposer.core.mps_backend import MpsBackend
            backend = MpsBackend()
        yield

    app = FastAPI(lifespan=lifespan)
    app.state.store = store

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> Any:
        return templates.TemplateResponse(request, "index.html", {})

    @app.post("/jobs")
    async def post_job(req: JobRequest) -> dict:
        img_path = ASSETS / req.sample
        if not img_path.exists():
            raise HTTPException(404, f"sample not found: {req.sample}")
        job = store.create()
        asyncio.create_task(_run_job(job.id, img_path, req, store, backend, inference_lock))
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

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(job.output_dir.iterdir()):
                z.write(f, arcname=f.name)
        buf.seek(0)
        return StreamingResponse(buf, media_type="application/zip",
                                 headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'})

    return app


async def _run_job(job_id: str, img_path: Path, req: JobRequest, store: JobStore,
                    backend: InferenceBackend, lock: asyncio.Lock) -> None:
    job = store.get(job_id)
    if job is None:
        return
    out_dir = Path("runs") / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    job.output_dir = out_dir
    tracer = Tracer(run_id=job_id)

    async def fwd(event: str, payload: dict) -> None:
        await store.publish(job_id, event, payload)

    def listener(event: str, payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(fwd(event, payload), loop)

    loop = asyncio.get_running_loop()
    tracer.subscribe(listener)

    job.status = "running"
    try:
        async with lock:
            img = Image.open(img_path).convert("RGB")
            layers = await asyncio.wait_for(
                asyncio.to_thread(
                    backend.decompose, img, req.layers, req.resolution, req.steps, None, tracer
                ),
                timeout=600.0,
            )
        for i, layer in enumerate(layers):
            layer.save(out_dir / f"layer_{i}.png")
        rep = tracer.report()
        (out_dir / "trace.json").write_text(report_to_json(rep))
        (out_dir / "trace.perfetto.json").write_text(json.dumps(to_perfetto(rep)))
        job.status = "done"
        await store.publish(job_id, "done", {"layers": len(layers)})
    except asyncio.TimeoutError:
        job.status = "error"
        job.error = "inference exceeded 600s wall-time budget"
        await store.publish(job_id, "error", {"message": job.error})
    except (RuntimeError, MemoryError) as e:
        msg = str(e)
        kind = "OOM" if ("out of memory" in msg.lower() or isinstance(e, MemoryError)) else "RuntimeError"
        job.status = "error"
        job.error = f"{kind}: {msg}"
        await store.publish(job_id, "error", {"message": job.error, "kind": kind})
    except Exception as e:
        job.status = "error"
        job.error = str(e)
        await store.publish(job_id, "error", {"message": str(e)})


app = create_app()
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_web_app.py -v`
Expected: 3 passed

- [ ] **Step 5: Manual smoke (with fake backend)**

Run in one shell: `uv run python -c "from decomposer.web.app import create_app; from decomposer.core.backend import FakeBackend; import uvicorn; uvicorn.run(create_app(backend=FakeBackend(latency_ms=200)), port=8000)"`

In another shell, open `http://localhost:8000`. Pick a sample, click Decompose, watch the live log, download the ZIP.
Expected: page renders, SSE events stream, ZIP downloads.

- [ ] **Step 6: Commit**

```bash
git add decomposer/web/ tests/test_web_app.py
git commit -m "Add FastAPI app: lifespan, POST /jobs, SSE, ZIP, index page"
```

---

## Task 14: Bundle test assets & wire up end-to-end

**Files:**
- Create: `decomposer/test_assets/sample_ad_1.png`, `sample_ad_2.png`, `tiny_smoke_test.png`
- Create: `README.md`

- [ ] **Step 1: Generate placeholder test images**

Run:
```bash
uv run python -c "
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
d = Path('decomposer/test_assets'); d.mkdir(parents=True, exist_ok=True)
for name, size, color, text in [
    ('sample_ad_1.png', (1024, 1024), (200, 60, 60), 'SUMMER SALE 50% OFF'),
    ('sample_ad_2.png', (1024, 1024), (40, 80, 200), 'NEW ARRIVAL'),
    ('tiny_smoke_test.png', (256, 256), (180, 80, 80), 'TEST'),
]:
    img = Image.new('RGB', size, color)
    draw = ImageDraw.Draw(img)
    draw.text((size[0]//4, size[1]//2 - 20), text, fill='white')
    img.save(d / name)
print('wrote 3 images')
"
```
Expected: `wrote 3 images`

Replace these with real marketing samples when you have them — the design works on whatever you put in `test_assets/`.

- [ ] **Step 2: Create `README.md`**

```markdown
# decomposer

Local Qwen-Image-Layered marketing-asset decomposer.

## Setup

```bash
uv sync --extra dev
uv run decomposer doctor --fake    # validate install
uv run decomposer doctor           # validate MPS path (downloads ~50 GB first time)
```

## Usage

CLI:
```bash
uv run decomposer decompose path/to/image.png --layers 6 --out ./out/
uv run decomposer diff-traces runs/<a>/trace.json runs/<b>/trace.json
```

Web:
```bash
uv run uvicorn decomposer.web.app:app --host 127.0.0.1 --port 8000
```

Then open <http://localhost:8000>.

See `docs/superpowers/specs/2026-05-17-decomposer-design.md` for full design.
```

- [ ] **Step 3: Run the full unit test suite**

Run: `uv run pytest -m "not mps_required" -v`
Expected: all non-MPS tests pass.

- [ ] **Step 4: Run ruff**

Run: `uv run ruff check . && uv run ruff format --check .`
Expected: clean (or fix what it complains about).

- [ ] **Step 5: Commit**

```bash
git add decomposer/test_assets/ README.md
git commit -m "Add bundled test assets and README"
```

- [ ] **Step 6: Final integration check (MPS only)**

If on M3 Max:

```bash
uv run decomposer doctor
```
Expected: passes; X-ray report printed; 3 layers from `tiny_smoke_test.png` (~30–60 s after model download).

Then:
```bash
uv run uvicorn decomposer.web.app:app --host 127.0.0.1 --port 8000
```
Open <http://localhost:8000>, pick `tiny_smoke_test.png`, click Decompose, wait, download ZIP, inspect the layers and `trace.perfetto.json` in `ui.perfetto.dev`.

If any stage in the live log fails, the X-ray report in `runs/<job_id>/trace.json` identifies which one — that's the diagnostic surface the X-ray layer was built for.

---

## Done

All spec requirements implemented:
- Five-phase inference pipeline with sequential residency ✓
- X-ray Tracer with JSON + Perfetto export ✓
- CPU-fallback detection ✓
- Q8 GGUF dequant module ✓
- CLI: doctor, decompose, diff-traces ✓
- FastAPI: lifespan, POST /jobs, SSE, ZIP, index page ✓
- Bundled test assets ✓
- Unit tests + MPS-gated integration tests ✓
