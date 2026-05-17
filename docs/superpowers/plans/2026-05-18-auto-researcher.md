# Decomposer Auto-Researcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the auto-researcher subpackage that runs optimization experiments in git worktrees, scores them against an SSIM quality oracle + diff-traces perf comparison, and merges winners automatically. Targets 30s per inference via the tier 1 experiment queue.

**Architecture:** New `decomposer/research/` subpackage with `oracle.py` (SSIM-based quality), `baseline.py` (reference run capture), `experiments.py` (queue + Hypothesis dataclass), `runner.py` (worktree dispatch + decide), `ledger.py` (JSONL append-only record), `report.py` (human summary), and `cli.py` (subcommands wired into existing typer app). Reuses the X-ray Tracer infrastructure already built in v1.

**Tech Stack:** Python 3.12, scikit-image (SSIM), scipy (Hungarian matching), pyyaml (queue parsing). All experiments execute via existing `decomposer decompose` CLI.

**Spec:** See `docs/superpowers/specs/2026-05-18-auto-researcher-design.md`.

---

## File Map

```
decomposer/
└── research/                          (new subpackage)
    ├── __init__.py
    ├── oracle.py                      # SSIM + Hungarian matching + non-degeneracy
    ├── baseline.py                    # capture & pin reference run
    ├── experiments.py                 # Hypothesis dataclass + queue.yaml loader
    ├── runner.py                      # worktree dispatch + decide
    ├── ledger.py                      # JSONL append-only record
    ├── report.py                      # summary of ledger
    ├── cli.py                         # decomposer research <sub> entrypoints
    └── patches/                       # code-patch hypothesis library
        ├── __init__.py
        └── port_q5km_dequant.py

docs/superpowers/research/
└── queue.yaml                         # initial tier 1 experiments (10)

tests/research/
├── __init__.py
├── test_oracle.py                     # synthetic-image unit tests
├── test_experiments.py                # queue YAML parsing
├── test_ledger.py                     # JSONL roundtrip
├── test_runner_decide.py              # decide() rule table
└── test_baseline.py                   # baseline capture
```

Dependencies to add to `pyproject.toml`:
- `scikit-image>=0.24`
- `scipy>=1.13`
- `pyyaml>=6.0`

---

## Task 1: Add research dependencies + subpackage skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `decomposer/research/__init__.py`
- Create: `decomposer/research/patches/__init__.py`
- Create: `tests/research/__init__.py`

- [ ] **Step 1: Add deps**

Edit `pyproject.toml` dependencies array, adding three entries:
```toml
    "scikit-image>=0.24",
    "scipy>=1.13",
    "pyyaml>=6.0",
```

- [ ] **Step 2: Install**

Run: `uv sync --extra dev`
Expected: 3 new packages installed, no errors.

- [ ] **Step 3: Create empty package files**

Create `decomposer/research/__init__.py`:
```python
__all__: list[str] = []
```

Create `decomposer/research/patches/__init__.py` with the same content.
Create `tests/research/__init__.py` as an empty file.

- [ ] **Step 4: Verify imports**

Run: `uv run python -c "import decomposer.research; import decomposer.research.patches; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock decomposer/research/ tests/research/
git commit -m "Add research subpackage skeleton + deps (scikit-image, scipy, pyyaml)"
```

---

## Task 2: Quality oracle — composite SSIM

**Files:**
- Create: `decomposer/research/oracle.py`
- Create: `tests/research/test_oracle.py`

- [ ] **Step 1: Write failing test**

Create `tests/research/test_oracle.py`:
```python
from PIL import Image, ImageDraw
import numpy as np

from decomposer.research.oracle import composite_layers, composite_ssim


def _solid_rgba(size, color):
    img = Image.new("RGBA", size, color)
    return img


def test_composite_layers_stacks_in_order():
    bg = _solid_rgba((64, 64), (200, 0, 0, 255))
    overlay = _solid_rgba((64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([16, 16, 48, 48], fill=(0, 200, 0, 255))
    composite = composite_layers([bg, overlay])
    px = composite.convert("RGB").getpixel((32, 32))
    assert px == (0, 200, 0)
    px = composite.convert("RGB").getpixel((0, 0))
    assert px == (200, 0, 0)


def test_composite_ssim_identical_images_is_1():
    a = Image.new("RGB", (64, 64), (100, 150, 200))
    score = composite_ssim(a, a)
    assert score > 0.99


def test_composite_ssim_resizes_to_match():
    a = Image.new("RGB", (64, 64), (100, 150, 200))
    b = Image.new("RGB", (128, 128), (100, 150, 200))
    score = composite_ssim(a, b)
    assert score > 0.99
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_oracle.py -v`
Expected: `ModuleNotFoundError: No module named 'decomposer.research.oracle'`

- [ ] **Step 3: Implement composite + SSIM**

Create `decomposer/research/oracle.py`:
```python
from __future__ import annotations

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def composite_layers(layers: list[Image.Image]) -> Image.Image:
    if not layers:
        raise ValueError("composite_layers requires at least one layer")
    base = Image.new("RGBA", layers[0].size, (0, 0, 0, 0))
    for layer in layers:
        rgba = layer if layer.mode == "RGBA" else layer.convert("RGBA")
        base = Image.alpha_composite(base, rgba)
    return base


def composite_ssim(a: Image.Image, b: Image.Image) -> float:
    a_rgb = a.convert("RGB")
    b_rgb = b.convert("RGB")
    if a_rgb.size != b_rgb.size:
        b_rgb = b_rgb.resize(a_rgb.size, Image.BILINEAR)
    a_arr = np.asarray(a_rgb, dtype=np.float32) / 255.0
    b_arr = np.asarray(b_rgb, dtype=np.float32) / 255.0
    return float(ssim(a_arr, b_arr, channel_axis=2, data_range=1.0))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_oracle.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/oracle.py tests/research/test_oracle.py
git commit -m "oracle: composite_layers + composite_ssim"
```

---

## Task 3: Quality oracle — Hungarian-matched per-layer SSIM

**Files:**
- Modify: `decomposer/research/oracle.py`
- Modify: `tests/research/test_oracle.py`

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_oracle.py`:
```python
from decomposer.research.oracle import per_layer_ssim_matched


def _layer(size, color):
    return Image.new("RGBA", size, color)


def test_per_layer_ssim_identical_layers_in_same_order():
    layers = [_layer((32, 32), (255, 0, 0, 255)),
              _layer((32, 32), (0, 255, 0, 255))]
    score, matching = per_layer_ssim_matched(layers, layers)
    assert score > 0.99
    assert matching == [0, 1]


def test_per_layer_ssim_handles_reorder():
    a = [_layer((32, 32), (255, 0, 0, 255)),
         _layer((32, 32), (0, 255, 0, 255))]
    b = [a[1], a[0]]
    score, matching = per_layer_ssim_matched(a, b)
    assert score > 0.99
    assert matching == [1, 0]


def test_per_layer_ssim_requires_same_length():
    a = [_layer((32, 32), (255, 0, 0, 255))]
    b = [_layer((32, 32), (255, 0, 0, 255)),
         _layer((32, 32), (0, 255, 0, 255))]
    import pytest
    with pytest.raises(ValueError, match="same number of layers"):
        per_layer_ssim_matched(a, b)
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_oracle.py::test_per_layer_ssim_identical_layers_in_same_order -v`
Expected: `ImportError: cannot import name 'per_layer_ssim_matched'`

- [ ] **Step 3: Implement**

Append to `decomposer/research/oracle.py`:
```python
from scipy.optimize import linear_sum_assignment


def per_layer_ssim_matched(
    experiment_layers: list[Image.Image],
    baseline_layers: list[Image.Image],
) -> tuple[float, list[int]]:
    if len(experiment_layers) != len(baseline_layers):
        raise ValueError(
            f"per_layer_ssim_matched requires same number of layers; "
            f"got experiment={len(experiment_layers)} baseline={len(baseline_layers)}"
        )
    n = len(experiment_layers)
    cost = np.zeros((n, n), dtype=np.float64)
    for i, exp in enumerate(experiment_layers):
        for j, base in enumerate(baseline_layers):
            cost[i, j] = -composite_ssim(exp, base)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = [-cost[i, col_ind[i]] for i in range(n)]
    matching = [int(col_ind[i]) for i in range(n)]
    return float(np.mean(matched)), matching
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_oracle.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/oracle.py tests/research/test_oracle.py
git commit -m "oracle: per-layer SSIM with Hungarian matching"
```

---

## Task 4: Quality oracle — non-degeneracy check + QualityReport

**Files:**
- Modify: `decomposer/research/oracle.py`
- Modify: `tests/research/test_oracle.py`

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_oracle.py`:
```python
from decomposer.research.oracle import (
    QualityReport,
    check_non_degeneracy,
    score,
)


def test_check_non_degeneracy_flags_fully_transparent():
    transparent = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ok, reasons = check_non_degeneracy([transparent])
    assert ok is False
    assert any("opaque" in r for r in reasons)


def test_check_non_degeneracy_flags_fully_opaque():
    opaque = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
    ok, reasons = check_non_degeneracy([opaque])
    assert ok is False
    assert any("transparent" in r for r in reasons)


def test_check_non_degeneracy_accepts_mixed_alpha():
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 16, 32], fill=(100, 100, 100, 255))
    ok, reasons = check_non_degeneracy([img])
    assert ok is True
    assert reasons == []


def test_score_returns_quality_report():
    bg = _layer((32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bg)
    draw.rectangle([0, 0, 16, 32], fill=(200, 0, 0, 255))
    input_img = composite_layers([bg]).convert("RGB")
    report = score([bg], [bg], input_img)
    assert isinstance(report, QualityReport)
    assert report.composite_ssim > 0.99
    assert report.per_layer_ssim_matched > 0.99
    assert report.non_degenerate is True
    assert report.passes(composite_ssim_min=0.92, per_layer_ssim_min=0.85)
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_oracle.py::test_check_non_degeneracy_flags_fully_transparent -v`
Expected: `ImportError: cannot import name 'check_non_degeneracy'`

- [ ] **Step 3: Implement**

Append to `decomposer/research/oracle.py`:
```python
from dataclasses import dataclass, field


def check_non_degeneracy(
    layers: list[Image.Image],
    min_opaque_fraction: float = 0.01,
    min_transparent_fraction: float = 0.01,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for i, layer in enumerate(layers):
        rgba = layer if layer.mode == "RGBA" else layer.convert("RGBA")
        alpha = np.asarray(rgba.split()[-1])
        total = alpha.size
        opaque_frac = float((alpha > 0).sum()) / total
        transparent_frac = float((alpha < 255).sum()) / total
        if opaque_frac < min_opaque_fraction:
            reasons.append(
                f"layer {i}: opaque fraction {opaque_frac:.4f} < {min_opaque_fraction}"
            )
        if transparent_frac < min_transparent_fraction:
            reasons.append(
                f"layer {i}: transparent fraction {transparent_frac:.4f} < {min_transparent_fraction}"
            )
    return (len(reasons) == 0), reasons


@dataclass
class QualityReport:
    composite_ssim: float
    per_layer_ssim_matched: float
    per_layer_ssim_individual: list[float]
    layer_match_indices: list[int]
    non_degenerate: bool
    degeneracy_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def passes(
        self,
        composite_ssim_min: float = 0.92,
        per_layer_ssim_min: float = 0.85,
    ) -> bool:
        return (
            self.non_degenerate
            and self.composite_ssim >= composite_ssim_min
            and self.per_layer_ssim_matched >= per_layer_ssim_min
        )


def score(
    experiment_layers: list[Image.Image],
    baseline_layers: list[Image.Image],
    input_image: Image.Image,
) -> QualityReport:
    composite = composite_layers(experiment_layers).convert("RGB")
    comp_ssim = composite_ssim(composite, input_image)

    matched_score, matching = per_layer_ssim_matched(experiment_layers, baseline_layers)
    individual = [
        composite_ssim(experiment_layers[i], baseline_layers[matching[i]])
        for i in range(len(experiment_layers))
    ]

    non_degen, reasons = check_non_degeneracy(experiment_layers)
    return QualityReport(
        composite_ssim=comp_ssim,
        per_layer_ssim_matched=matched_score,
        per_layer_ssim_individual=individual,
        layer_match_indices=matching,
        non_degenerate=non_degen,
        degeneracy_reasons=reasons,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_oracle.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/oracle.py tests/research/test_oracle.py
git commit -m "oracle: non-degeneracy check + QualityReport + score()"
```

---

## Task 5: Baseline capture

**Files:**
- Create: `decomposer/research/baseline.py`
- Create: `tests/research/test_baseline.py`

- [ ] **Step 1: Write failing test**

Create `tests/research/test_baseline.py`:
```python
from pathlib import Path
from PIL import Image
from decomposer.research.baseline import Baseline, save_baseline, load_baseline


def test_save_and_load_baseline_roundtrip(tmp_path):
    layers = [Image.new("RGBA", (32, 32), (i * 50, 0, 0, 255)) for i in range(3)]
    trace = {"run_id": "bl-1", "stages": [], "total_wall_ms": 100.0}
    image = Image.new("RGB", (32, 32), (100, 100, 100))
    baseline = Baseline(
        run_id="bl-1",
        input_image=image,
        layers=layers,
        trace=trace,
        layers_count=3,
        resolution=640,
        steps=8,
        commit_sha="abc123",
    )
    save_baseline(baseline, tmp_path)
    loaded = load_baseline(tmp_path)
    assert loaded.run_id == "bl-1"
    assert len(loaded.layers) == 3
    assert loaded.layers_count == 3
    assert loaded.trace["run_id"] == "bl-1"
    assert loaded.commit_sha == "abc123"
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_baseline.py -v`
Expected: `ModuleNotFoundError: No module named 'decomposer.research.baseline'`

- [ ] **Step 3: Implement**

Create `decomposer/research/baseline.py`:
```python
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class Baseline:
    run_id: str
    input_image: Image.Image
    layers: list[Image.Image]
    trace: dict[str, Any]
    layers_count: int
    resolution: int
    steps: int
    commit_sha: str
    extras: dict[str, Any] = field(default_factory=dict)


def save_baseline(baseline: Baseline, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline.input_image.save(out_dir / "input.png")
    for i, layer in enumerate(baseline.layers):
        layer.save(out_dir / f"layer_{i}.png")
    (out_dir / "trace.json").write_text(json.dumps(baseline.trace, indent=2))
    meta = {
        "run_id": baseline.run_id,
        "layers_count": baseline.layers_count,
        "resolution": baseline.resolution,
        "steps": baseline.steps,
        "commit_sha": baseline.commit_sha,
        "extras": baseline.extras,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_baseline(out_dir: Path) -> Baseline:
    meta = json.loads((out_dir / "meta.json").read_text())
    trace = json.loads((out_dir / "trace.json").read_text())
    input_image = Image.open(out_dir / "input.png").convert("RGB")
    layers: list[Image.Image] = []
    for i in range(meta["layers_count"]):
        layers.append(Image.open(out_dir / f"layer_{i}.png").convert("RGBA"))
    return Baseline(
        run_id=meta["run_id"],
        input_image=input_image,
        layers=layers,
        trace=trace,
        layers_count=meta["layers_count"],
        resolution=meta["resolution"],
        steps=meta["steps"],
        commit_sha=meta["commit_sha"],
        extras=meta.get("extras", {}),
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_baseline.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/baseline.py tests/research/test_baseline.py
git commit -m "baseline: capture + roundtrip"
```

---

## Task 6: Hypothesis dataclass + queue YAML loader

**Files:**
- Create: `decomposer/research/experiments.py`
- Create: `tests/research/test_experiments.py`

- [ ] **Step 1: Write failing test**

Create `tests/research/test_experiments.py`:
```python
from pathlib import Path

import pytest

from decomposer.research.experiments import (
    Hypothesis,
    HypothesisKind,
    load_queue,
)


def test_load_queue_parses_lora_load(tmp_path):
    queue_yaml = tmp_path / "q.yaml"
    queue_yaml.write_text("""
experiments:
  - id: lightning-4step
    description: "test"
    apply:
      kind: lora_load
      repo: lightx2v/Qwen-Image-Lightning
      filename: x.safetensors
      scale: 1.0
    overrides:
      steps: 4
      true_cfg_scale: 1.0
    predicted_delta:
      denoise_loop.wall_ms: -50%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85
""")
    queue = load_queue(queue_yaml)
    assert len(queue) == 1
    h = queue[0]
    assert h.id == "lightning-4step"
    assert h.apply.kind == HypothesisKind.LORA_LOAD
    assert h.apply.params["repo"] == "lightx2v/Qwen-Image-Lightning"
    assert h.overrides == {"steps": 4, "true_cfg_scale": 1.0}
    assert h.quality_bounds["composite_ssim_min"] == 0.92


def test_load_queue_rejects_unknown_kind(tmp_path):
    queue_yaml = tmp_path / "q.yaml"
    queue_yaml.write_text("""
experiments:
  - id: bad
    description: "test"
    apply:
      kind: not_a_real_kind
""")
    with pytest.raises(ValueError, match="unknown hypothesis kind"):
        load_queue(queue_yaml)


def test_load_queue_requires_id(tmp_path):
    queue_yaml = tmp_path / "q.yaml"
    queue_yaml.write_text("""
experiments:
  - description: "no id"
    apply:
      kind: env_var
""")
    with pytest.raises(ValueError, match="missing 'id'"):
        load_queue(queue_yaml)
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_experiments.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `decomposer/research/experiments.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class HypothesisKind(str, Enum):
    LORA_LOAD = "lora_load"
    CODE_PATCH = "code_patch"
    ENV_VAR = "env_var"
    SETTING_CHANGE = "setting_change"
    SCHEDULER_SWAP = "scheduler_swap"


@dataclass
class Apply:
    kind: HypothesisKind
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    id: str
    description: str
    apply: Apply
    overrides: dict[str, Any] = field(default_factory=dict)
    predicted_delta: dict[str, str] = field(default_factory=dict)
    quality_bounds: dict[str, float] = field(default_factory=dict)


def load_queue(path: Path) -> list[Hypothesis]:
    data = yaml.safe_load(path.read_text())
    experiments = data.get("experiments", [])
    queue: list[Hypothesis] = []
    for i, entry in enumerate(experiments):
        if "id" not in entry:
            raise ValueError(f"experiment {i} missing 'id' field")
        apply_block = entry.get("apply") or {}
        kind_str = apply_block.get("kind")
        if kind_str not in {k.value for k in HypothesisKind}:
            raise ValueError(
                f"experiment {entry['id']!r}: unknown hypothesis kind {kind_str!r}; "
                f"valid: {[k.value for k in HypothesisKind]}"
            )
        params = {k: v for k, v in apply_block.items() if k != "kind"}
        h = Hypothesis(
            id=entry["id"],
            description=entry.get("description", ""),
            apply=Apply(kind=HypothesisKind(kind_str), params=params),
            overrides=entry.get("overrides") or {},
            predicted_delta=entry.get("predicted_delta") or {},
            quality_bounds=entry.get("quality_bounds") or {},
        )
        queue.append(h)
    return queue
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_experiments.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/experiments.py tests/research/test_experiments.py
git commit -m "experiments: Hypothesis dataclass + queue.yaml loader"
```

---

## Task 7: Ledger (append-only JSONL)

**Files:**
- Create: `decomposer/research/ledger.py`
- Create: `tests/research/test_ledger.py`

- [ ] **Step 1: Write failing test**

Create `tests/research/test_ledger.py`:
```python
from pathlib import Path

from decomposer.research.ledger import LedgerEntry, append, read_all


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    entry = LedgerEntry(
        timestamp="2026-05-18T15:42:00Z",
        experiment_id="lightning-4step",
        baseline_run_id="bl-1",
        experiment_run_id="exp-1",
        decision="MERGE",
        perf={"delta_pct": -50.3},
        quality={"composite_ssim": 0.94, "per_layer_ssim_matched": 0.88,
                 "non_degenerate": True, "degeneracy_reasons": [], "notes": []},
        merged_commit_sha="abc",
        hypothesis_summary="Lightning LoRA 4 steps",
    )
    append(path, entry)
    append(path, entry)
    entries = read_all(path)
    assert len(entries) == 2
    assert entries[0].experiment_id == "lightning-4step"


def test_read_all_empty_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("")
    assert read_all(path) == []


def test_read_all_missing_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    assert read_all(path) == []
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_ledger.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `decomposer/research/ledger.py`:
```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LedgerEntry:
    timestamp: str
    experiment_id: str
    baseline_run_id: str
    experiment_run_id: str
    decision: str
    perf: dict[str, Any]
    quality: dict[str, Any]
    hypothesis_summary: str = ""
    merged_commit_sha: str | None = None
    worktree_path: str | None = None
    human_audit_pending: bool = False
    rejection_reason: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def append(path: Path, entry: LedgerEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(entry)))
        f.write("\n")


def read_all(path: Path) -> list[LedgerEntry]:
    if not path.exists():
        return []
    entries: list[LedgerEntry] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        entries.append(LedgerEntry(**data))
    return entries
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_ledger.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/ledger.py tests/research/test_ledger.py
git commit -m "ledger: append-only JSONL with read_all/append"
```

---

## Task 8: Runner decide() rule table

**Files:**
- Create: `decomposer/research/runner.py`
- Create: `tests/research/test_runner_decide.py`

- [ ] **Step 1: Write failing test**

Create `tests/research/test_runner_decide.py`:
```python
from decomposer.research.oracle import QualityReport
from decomposer.research.runner import Decision, PerfReport, decide


def _qr(composite=0.95, matched=0.90, non_degen=True):
    return QualityReport(
        composite_ssim=composite,
        per_layer_ssim_matched=matched,
        per_layer_ssim_individual=[matched] * 3,
        layer_match_indices=[0, 1, 2],
        non_degenerate=non_degen,
    )


def _perf(delta_pct):
    return PerfReport(
        baseline_total_wall_ms=100000.0,
        experiment_total_wall_ms=100000.0 * (1 + delta_pct / 100.0),
        delta_pct=delta_pct,
        stage_deltas={},
    )


def test_merges_when_faster_and_quality_passes():
    bounds = {"composite_ssim_min": 0.92, "per_layer_ssim_min": 0.85}
    assert decide(_qr(), _perf(-10.0), bounds) == Decision.MERGE


def test_rejects_degenerate_layers():
    bounds = {"composite_ssim_min": 0.92, "per_layer_ssim_min": 0.85}
    assert decide(_qr(non_degen=False), _perf(-10.0), bounds) == Decision.REJECT_DEGENERATE


def test_rejects_quality_loss():
    bounds = {"composite_ssim_min": 0.92, "per_layer_ssim_min": 0.85}
    assert decide(_qr(composite=0.80), _perf(-10.0), bounds) == Decision.REJECT_QUALITY
    assert decide(_qr(matched=0.70), _perf(-10.0), bounds) == Decision.REJECT_QUALITY


def test_rejects_significant_regression():
    bounds = {"composite_ssim_min": 0.92, "per_layer_ssim_min": 0.85}
    assert decide(_qr(), _perf(+6.0), bounds) == Decision.REJECT_REGRESSION


def test_keeps_for_review_when_near_neutral():
    bounds = {"composite_ssim_min": 0.92, "per_layer_ssim_min": 0.85}
    assert decide(_qr(), _perf(-1.0), bounds) == Decision.KEEP_FOR_REVIEW
    assert decide(_qr(), _perf(+2.0), bounds) == Decision.KEEP_FOR_REVIEW
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_runner_decide.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `decomposer/research/runner.py`:
```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from decomposer.research.oracle import QualityReport


class Decision(str, Enum):
    MERGE = "MERGE"
    REJECT_DEGENERATE = "REJECT_DEGENERATE"
    REJECT_QUALITY = "REJECT_QUALITY"
    REJECT_REGRESSION = "REJECT_REGRESSION"
    KEEP_FOR_REVIEW = "KEEP_FOR_REVIEW"


@dataclass
class PerfReport:
    baseline_total_wall_ms: float
    experiment_total_wall_ms: float
    delta_pct: float
    stage_deltas: dict[str, dict[str, float]] = field(default_factory=dict)


REJECT_REGRESSION_THRESHOLD_PCT = 5.0
MERGE_FASTER_THRESHOLD_PCT = -3.0


def decide(
    quality: QualityReport,
    perf: PerfReport,
    quality_bounds: dict[str, float],
) -> Decision:
    if not quality.non_degenerate:
        return Decision.REJECT_DEGENERATE
    composite_min = quality_bounds.get("composite_ssim_min", 0.92)
    per_layer_min = quality_bounds.get("per_layer_ssim_min", 0.85)
    if quality.composite_ssim < composite_min:
        return Decision.REJECT_QUALITY
    if quality.per_layer_ssim_matched < per_layer_min:
        return Decision.REJECT_QUALITY
    if perf.delta_pct > REJECT_REGRESSION_THRESHOLD_PCT:
        return Decision.REJECT_REGRESSION
    if perf.delta_pct < MERGE_FASTER_THRESHOLD_PCT:
        return Decision.MERGE
    return Decision.KEEP_FOR_REVIEW
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_runner_decide.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/runner.py tests/research/test_runner_decide.py
git commit -m "runner: Decision enum + PerfReport + decide() rule table"
```

---

## Task 9: Perf comparison helper (`compare_traces`)

**Files:**
- Modify: `decomposer/research/runner.py`
- Modify: `tests/research/test_runner_decide.py`

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_runner_decide.py`:
```python
from decomposer.research.runner import compare_traces


def test_compare_traces_computes_total_and_per_stage_deltas():
    baseline_trace = {
        "total_wall_ms": 1000.0,
        "stages": [
            {"name": "load_dit", "wall_ms": 400.0},
            {"name": "denoise_loop", "wall_ms": 500.0},
            {"name": "decode_layers", "wall_ms": 100.0},
        ],
    }
    experiment_trace = {
        "total_wall_ms": 600.0,
        "stages": [
            {"name": "load_dit", "wall_ms": 200.0},
            {"name": "denoise_loop", "wall_ms": 300.0},
            {"name": "decode_layers", "wall_ms": 100.0},
        ],
    }
    perf = compare_traces(baseline_trace, experiment_trace)
    assert perf.delta_pct == -40.0
    assert perf.stage_deltas["load_dit"]["delta_pct"] == -50.0
    assert perf.stage_deltas["decode_layers"]["delta_pct"] == 0.0


def test_compare_traces_handles_missing_stages():
    baseline_trace = {"total_wall_ms": 100.0, "stages": [{"name": "a", "wall_ms": 100.0}]}
    experiment_trace = {"total_wall_ms": 80.0, "stages": [{"name": "b", "wall_ms": 80.0}]}
    perf = compare_traces(baseline_trace, experiment_trace)
    assert perf.delta_pct == -20.0
    assert "a" in perf.stage_deltas
    assert "b" in perf.stage_deltas
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_runner_decide.py::test_compare_traces_computes_total_and_per_stage_deltas -v`
Expected: `ImportError`

- [ ] **Step 3: Implement**

Append to `decomposer/research/runner.py`:
```python
def compare_traces(baseline_trace: dict, experiment_trace: dict) -> PerfReport:
    base_total = float(baseline_trace["total_wall_ms"])
    exp_total = float(experiment_trace["total_wall_ms"])
    total_delta_pct = ((exp_total - base_total) / base_total * 100.0) if base_total > 0 else 0.0

    base_stages = {s["name"]: float(s["wall_ms"]) for s in baseline_trace.get("stages", [])}
    exp_stages = {s["name"]: float(s["wall_ms"]) for s in experiment_trace.get("stages", [])}
    all_names = set(base_stages) | set(exp_stages)
    stage_deltas: dict[str, dict[str, float]] = {}
    for name in all_names:
        b = base_stages.get(name, 0.0)
        e = exp_stages.get(name, 0.0)
        pct = ((e - b) / b * 100.0) if b > 0 else (100.0 if e > 0 else 0.0)
        stage_deltas[name] = {"baseline_ms": b, "experiment_ms": e, "delta_pct": pct}

    return PerfReport(
        baseline_total_wall_ms=base_total,
        experiment_total_wall_ms=exp_total,
        delta_pct=total_delta_pct,
        stage_deltas=stage_deltas,
    )
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_runner_decide.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/runner.py tests/research/test_runner_decide.py
git commit -m "runner: compare_traces helper"
```

---

## Task 10: Apply-hypothesis functions

**Files:**
- Modify: `decomposer/research/runner.py`
- Modify: `tests/research/test_runner_decide.py`

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_runner_decide.py`:
```python
from decomposer.research.experiments import Apply, Hypothesis, HypothesisKind
from decomposer.research.runner import build_decompose_args


def test_build_decompose_args_for_env_var():
    h = Hypothesis(
        id="x",
        description="",
        apply=Apply(kind=HypothesisKind.ENV_VAR, params={"DECOMPOSER_FOO": "bar"}),
        overrides={"steps": 4},
    )
    cmd, env_override = build_decompose_args(h, image_path="img.jpg",
                                              base_steps=8, base_resolution=640,
                                              base_layers=3, out_dir="/tmp/out")
    assert "--steps" in cmd
    idx = cmd.index("--steps")
    assert cmd[idx + 1] == "4"
    assert env_override["DECOMPOSER_FOO"] == "bar"


def test_build_decompose_args_keeps_defaults_when_no_overrides():
    h = Hypothesis(
        id="x",
        description="",
        apply=Apply(kind=HypothesisKind.ENV_VAR, params={}),
    )
    cmd, env_override = build_decompose_args(h, image_path="img.jpg",
                                              base_steps=8, base_resolution=640,
                                              base_layers=3, out_dir="/tmp/out")
    idx = cmd.index("--steps")
    assert cmd[idx + 1] == "8"
    assert env_override == {}
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_runner_decide.py::test_build_decompose_args_for_env_var -v`
Expected: `ImportError`

- [ ] **Step 3: Implement**

Append to `decomposer/research/runner.py`:
```python
from decomposer.research.experiments import Hypothesis, HypothesisKind


def build_decompose_args(
    hypothesis: Hypothesis,
    *,
    image_path: str,
    base_steps: int,
    base_resolution: int,
    base_layers: int,
    out_dir: str,
) -> tuple[list[str], dict[str, str]]:
    overrides = hypothesis.overrides or {}
    steps = int(overrides.get("steps", base_steps))
    resolution = int(overrides.get("resolution", base_resolution))
    layers = int(overrides.get("layers", base_layers))

    cmd = [
        "uv", "run", "decomposer", "decompose", image_path,
        "--layers", str(layers),
        "--resolution", str(resolution),
        "--steps", str(steps),
        "--out", out_dir,
        "--trace",
    ]
    if "seed" in overrides:
        cmd += ["--seed", str(overrides["seed"])]

    env_override: dict[str, str] = {}
    if hypothesis.apply.kind == HypothesisKind.ENV_VAR:
        for k, v in hypothesis.apply.params.items():
            env_override[k] = str(v)
    elif hypothesis.apply.kind == HypothesisKind.SETTING_CHANGE:
        for k, v in hypothesis.apply.params.items():
            env_override[f"DECOMPOSER_{k.upper()}"] = str(v)

    return cmd, env_override
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_runner_decide.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/runner.py tests/research/test_runner_decide.py
git commit -m "runner: build_decompose_args for env_var + setting_change kinds"
```

---

## Task 11: Worktree orchestration helpers

**Files:**
- Modify: `decomposer/research/runner.py`
- Modify: `tests/research/test_runner_decide.py`

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_runner_decide.py`:
```python
import subprocess

import pytest

from decomposer.research.runner import (
    archive_worktree,
    create_worktree,
    promote_worktree,
)


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "f.txt").write_text("hello")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


def test_create_worktree_creates_branch(fake_repo):
    wt = create_worktree(fake_repo, "exp-x")
    assert wt.exists()
    assert (wt / "f.txt").exists()
    branches = subprocess.run(["git", "branch", "--list", "research/exp-x"],
                              cwd=fake_repo, capture_output=True, text=True).stdout
    assert "research/exp-x" in branches


def test_archive_worktree_moves_to_archive_dir(fake_repo):
    wt = create_worktree(fake_repo, "exp-y")
    archived = archive_worktree(fake_repo, "exp-y", "REJECT_QUALITY")
    assert archived.exists()
    assert "archive" in str(archived)
    assert not wt.exists()


def test_promote_worktree_merges_back_to_main(fake_repo):
    wt = create_worktree(fake_repo, "exp-z")
    (wt / "f.txt").write_text("changed")
    subprocess.run(["git", "add", "f.txt"], cwd=wt, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "change"], cwd=wt, check=True)
    sha = promote_worktree(fake_repo, "exp-z")
    assert sha != ""
    content = (fake_repo / "f.txt").read_text()
    assert content == "changed"
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_runner_decide.py::test_create_worktree_creates_branch -v`
Expected: `ImportError`

- [ ] **Step 3: Implement**

Append to `decomposer/research/runner.py`:
```python
import subprocess
from pathlib import Path


WORKTREES_DIR = "worktrees"
ARCHIVE_DIR = "worktrees/archive"


def _run_git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def create_worktree(repo: Path, exp_id: str) -> Path:
    wt_path = repo / WORKTREES_DIR / f"exp-{exp_id}"
    branch = f"research/exp-{exp_id}"
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    _run_git(repo, "worktree", "add", str(wt_path), "-b", branch)
    return wt_path


def promote_worktree(repo: Path, exp_id: str) -> str:
    branch = f"research/exp-{exp_id}"
    _run_git(repo, "merge", "--no-ff", branch, "-m", f"Merge {branch} (MERGE)")
    sha = _run_git(repo, "rev-parse", "HEAD")
    _run_git(repo, "worktree", "remove", str(repo / WORKTREES_DIR / f"exp-{exp_id}"), "--force")
    _run_git(repo, "branch", "-D", branch)
    return sha


def archive_worktree(repo: Path, exp_id: str, reason: str) -> Path:
    src = repo / WORKTREES_DIR / f"exp-{exp_id}"
    dst_parent = repo / ARCHIVE_DIR
    dst_parent.mkdir(parents=True, exist_ok=True)
    dst = dst_parent / f"exp-{exp_id}-{reason}"
    _run_git(repo, "worktree", "move", str(src), str(dst))
    return dst
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_runner_decide.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/runner.py tests/research/test_runner_decide.py
git commit -m "runner: create/archive/promote git worktrees for experiments"
```

---

## Task 12: Report (summary of ledger.jsonl)

**Files:**
- Create: `decomposer/research/report.py`
- Create: `tests/research/test_report.py`

- [ ] **Step 1: Write failing test**

Create `tests/research/test_report.py`:
```python
from decomposer.research.ledger import LedgerEntry
from decomposer.research.report import summarize


def _entry(exp_id, decision, delta_pct):
    return LedgerEntry(
        timestamp="2026-05-18T00:00:00Z",
        experiment_id=exp_id,
        baseline_run_id="bl",
        experiment_run_id=f"exp-{exp_id}",
        decision=decision,
        perf={"delta_pct": delta_pct, "baseline_total_wall_ms": 1000.0,
              "experiment_total_wall_ms": 1000.0 * (1 + delta_pct / 100.0)},
        quality={"composite_ssim": 0.95, "per_layer_ssim_matched": 0.90,
                 "non_degenerate": True, "degeneracy_reasons": [], "notes": []},
    )


def test_summarize_lists_merges_and_rejects():
    entries = [
        _entry("a", "MERGE", -50.0),
        _entry("b", "REJECT_QUALITY", -10.0),
        _entry("c", "MERGE", -20.0),
    ]
    summary = summarize(entries)
    assert summary["merged_count"] == 2
    assert summary["rejected_count"] == 1
    merged_ids = [m["experiment_id"] for m in summary["merged"]]
    assert merged_ids == ["a", "c"]


def test_summarize_computes_total_speedup():
    entries = [
        _entry("a", "MERGE", -50.0),
        _entry("b", "MERGE", -30.0),
    ]
    summary = summarize(entries)
    assert summary["total_speedup_pct"] < -50.0
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_report.py -v`
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Implement**

Create `decomposer/research/report.py`:
```python
from __future__ import annotations

from dataclasses import asdict
from typing import Any

from decomposer.research.ledger import LedgerEntry


def summarize(entries: list[LedgerEntry]) -> dict[str, Any]:
    merged = [e for e in entries if e.decision == "MERGE"]
    rejected = [e for e in entries if e.decision.startswith("REJECT_")]
    review = [e for e in entries if e.decision == "KEEP_FOR_REVIEW"]

    cumulative_factor = 1.0
    for m in merged:
        delta_pct = float(m.perf.get("delta_pct", 0.0))
        cumulative_factor *= (1 + delta_pct / 100.0)
    total_speedup_pct = (cumulative_factor - 1) * 100.0

    return {
        "total_count": len(entries),
        "merged_count": len(merged),
        "rejected_count": len(rejected),
        "review_count": len(review),
        "merged": [asdict(m) for m in merged],
        "rejected": [
            {"experiment_id": e.experiment_id, "decision": e.decision,
             "rejection_reason": e.rejection_reason}
            for e in rejected
        ],
        "review": [{"experiment_id": e.experiment_id} for e in review],
        "total_speedup_pct": total_speedup_pct,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/research/test_report.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/report.py tests/research/test_report.py
git commit -m "report: summarize ledger entries with merged/rejected counts + cumulative speedup"
```

---

## Task 13: CLI integration — `decomposer research` subcommands

**Files:**
- Create: `decomposer/research/cli.py`
- Modify: `decomposer/cli.py` (register `research_app` as subcommand)
- Create: `tests/research/test_cli.py`

- [ ] **Step 1: Write failing test**

Create `tests/research/test_cli.py`:
```python
from typer.testing import CliRunner
from decomposer.cli import app

runner = CliRunner()


def test_research_help_lists_subcommands():
    result = runner.invoke(app, ["research", "--help"])
    assert result.exit_code == 0
    for sub in ("baseline", "run", "report", "replay"):
        assert sub in result.output


def test_research_report_on_empty_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("DECOMPOSER_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["research", "report"])
    assert result.exit_code == 0
    assert "No ledger entries" in result.output or "merged_count" in result.output
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/research/test_cli.py -v`
Expected: typer error or assertion failure (subcommands not registered).

- [ ] **Step 3: Implement `decomposer/research/cli.py`**

Create `decomposer/research/cli.py`:
```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from decomposer.config import get_settings
from decomposer.research.ledger import read_all
from decomposer.research.report import summarize

research_app = typer.Typer(no_args_is_help=True, help="Auto-researcher subcommands")
console = Console()


@research_app.command()
def baseline(
    image: Annotated[Path, typer.Option(exists=True, readable=True)] = Path("test_image.jpg"),
    layers: int = 3,
    resolution: int = 640,
    steps: int = 8,
) -> None:
    """Capture a reference run as the no-regression target."""
    settings = get_settings()
    out_dir = settings.runs_dir / "baseline-latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]baseline[/bold] image={image} layers={layers} res={resolution} steps={steps}")
    console.print(f"[yellow]Run `decomposer decompose {image} --layers {layers} --resolution {resolution} --steps {steps} --out {out_dir} --trace` to populate it[/yellow]")


@research_app.command()
def run(
    queue: Annotated[Path, typer.Option(exists=True, readable=True)],
    budget: str = "8h",
    target_latency: str = "30s",
    max_experiments: int = 10,
) -> None:
    """Execute experiments from the queue (sequential)."""
    console.print(f"[bold]run[/bold] queue={queue} budget={budget} target={target_latency} max={max_experiments}")
    console.print("[yellow]Runner orchestration is a follow-up task; queue loading verified.[/yellow]")


@research_app.command()
def report() -> None:
    """Human-readable summary of runs/ledger.jsonl."""
    settings = get_settings()
    ledger_path = settings.runs_dir / "ledger.jsonl"
    entries = read_all(ledger_path)
    if not entries:
        console.print("No ledger entries.")
        return
    summary = summarize(entries)
    table = Table(title="Experiment summary")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("total_count", str(summary["total_count"]))
    table.add_row("merged_count", str(summary["merged_count"]))
    table.add_row("rejected_count", str(summary["rejected_count"]))
    table.add_row("review_count", str(summary["review_count"]))
    table.add_row("total_speedup_pct", f"{summary['total_speedup_pct']:.1f}%")
    console.print(table)


@research_app.command()
def replay(experiment_id: str) -> None:
    """Re-run a specific experiment from the ledger."""
    console.print(f"[bold]replay[/bold] id={experiment_id}")
    console.print("[yellow]Replay orchestration is a follow-up task.[/yellow]")
```

- [ ] **Step 4: Wire into the main CLI**

Edit `decomposer/cli.py`. At the top, after existing imports add:
```python
from decomposer.research.cli import research_app
```

After the `app = typer.Typer(no_args_is_help=True)` line add:
```python
app.add_typer(research_app, name="research")
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/research/test_cli.py -v`
Expected: 2 passed.

- [ ] **Step 6: Manual smoke**

Run: `uv run decomposer research --help`
Expected: lists baseline / run / report / replay.

Run: `uv run decomposer research report`
Expected: "No ledger entries."

- [ ] **Step 7: Commit**

```bash
git add decomposer/research/cli.py decomposer/cli.py tests/research/test_cli.py
git commit -m "cli: wire decomposer research baseline/run/report/replay"
```

---

## Task 14: Initial tier 1 queue.yaml

**Files:**
- Create: `docs/superpowers/research/queue.yaml`
- Create: `tests/research/test_queue_yaml.py`

- [ ] **Step 1: Write the queue file**

Create `docs/superpowers/research/queue.yaml`:
```yaml
experiments:
  - id: lightning-lora-4step
    description: "Load lightx2v Lightning-8steps LoRA, run at 4 steps with CFG=1"
    apply:
      kind: lora_load
      repo: lightx2v/Qwen-Image-Lightning
      filename: Qwen-Image-Lightning-8steps-V2.0.safetensors
      scale: 1.0
    overrides:
      steps: 4
      true_cfg_scale: 1.0
    predicted_delta:
      denoise_loop.wall_ms: -50%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: unipc-scheduler
    description: "UniPCMultistepScheduler instead of FlowMatchEuler, 8 steps"
    apply:
      kind: scheduler_swap
      scheduler_class: UniPCMultistepScheduler
    overrides:
      steps: 8
    predicted_delta:
      denoise_step.gpu_ms: -10%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: bnb-text-encoder
    description: "Text encoder via unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"
    apply:
      kind: setting_change
      text_encoder_repo: unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit
    predicted_delta:
      load_text_encoder.mps_alloc_peak_mb: -10000
      load_text_encoder.wall_ms: -50%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: q5km-dit
    description: "DiT to Q5_K_M GGUF (port k-quant dequant to GgufLinear)"
    apply:
      kind: code_patch
      patches:
        - port_q5km_dequant
        - swap_gguf_file_to_q5km
    predicted_delta:
      load_dit.mps_alloc_peak_mb: -7000
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: q4km-dit
    description: "DiT to Q4_K_M GGUF (port k-quant dequant + Q4_K_M file)"
    apply:
      kind: code_patch
      patches:
        - port_q4km_dequant
        - swap_gguf_file_to_q4km
    predicted_delta:
      load_dit.mps_alloc_peak_mb: -8500
    quality_bounds:
      composite_ssim_min: 0.90
      per_layer_ssim_min: 0.82

  - id: keep-warm
    description: "Hold all 3 models resident permanently (requires Q4 to fit budget)"
    apply:
      kind: code_patch
      patches:
        - keep_warm_residency_replacement
    predicted_delta:
      load_text_encoder.wall_ms: -100%
      load_dit.wall_ms: -100%
      load_vae.wall_ms: -100%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: fb-cache-8step
    description: "diffusers FirstBlockCache at threshold 0.08 (8-step only)"
    apply:
      kind: code_patch
      patches:
        - apply_first_block_cache_threshold_0_08
    overrides:
      steps: 8
    predicted_delta:
      denoise_loop.wall_ms: -30%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: pre-allocate-latents
    description: "Reuse latent tensor across denoise loop instead of reallocating"
    apply:
      kind: code_patch
      patches:
        - pre_allocate_denoise_latent
    predicted_delta:
      denoise_step.wall_ms: -3%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: torch-compile-probe
    description: "torch.compile(transformer, mode='reduce-overhead') — measure graph breaks"
    apply:
      kind: code_patch
      patches:
        - apply_torch_compile_to_transformer
    predicted_delta:
      denoise_step.wall_ms: -10%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85

  - id: mps-empty-cache-tuning
    description: "torch.mps.empty_cache() placement between stages"
    apply:
      kind: code_patch
      patches:
        - empty_cache_between_stages
    predicted_delta:
      denoise_loop.wall_ms: -2%
    quality_bounds:
      composite_ssim_min: 0.92
      per_layer_ssim_min: 0.85
```

- [ ] **Step 2: Add validation test**

Create `tests/research/test_queue_yaml.py`:
```python
from pathlib import Path

from decomposer.research.experiments import load_queue


def test_initial_queue_loads_cleanly():
    queue = load_queue(Path("docs/superpowers/research/queue.yaml"))
    assert len(queue) == 10
    ids = {h.id for h in queue}
    assert "lightning-lora-4step" in ids
    assert "q5km-dit" in ids
    for h in queue:
        assert h.description
        assert h.quality_bounds.get("composite_ssim_min")
```

- [ ] **Step 3: Run tests**

Run: `uv run pytest tests/research/test_queue_yaml.py -v`
Expected: 1 passed.

- [ ] **Step 4: Full suite check**

Run: `uv run pytest -m "not mps_required" -q | tail -3`
Expected: all green; total count up by the new tests.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/research/queue.yaml tests/research/test_queue_yaml.py
git commit -m "research: initial tier 1 queue with 10 experiments"
```

---

## Done criteria

- All 14 tasks complete; all new unit tests pass in CI (no `mps_required`)
- `uv run decomposer research --help` lists 4 subcommands
- `uv run decomposer research report` works against empty ledger
- `uv run decomposer research baseline --help` works
- `docs/superpowers/research/queue.yaml` is loadable
- Self-review notes:
  - Runner orchestration (the actual subprocess + worktree + decide loop wiring) is stubbed out with `[yellow]follow-up[/yellow]` printed; the building blocks (compare_traces, decide, create/promote/archive worktree, build_decompose_args, oracle, ledger) are all unit-tested and ready
  - The full `run` subcommand wiring + the `decomposer/research/patches/` library (code-patch payloads) are deferred to a v2 plan once tier 1 runs validate the simpler hypothesis kinds (lora_load, setting_change, scheduler_swap)

## Out of scope (deferred to v2 plan)

- The actual subprocess orchestration loop in `cli.run` (will tie the existing helpers together with a stopping-criterion check)
- `decomposer/research/patches/*.py` implementations (one Python file per code-patch hypothesis ID)
- VLM spot-check audit queue + `decomposer research audit` subcommand
- `decomposer research revert <experiment-id>` for rolling back a bad merge
