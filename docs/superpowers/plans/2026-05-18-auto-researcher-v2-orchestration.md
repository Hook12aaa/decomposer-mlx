# Auto-Researcher v2: Orchestration Loop + First Code Patches

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Wire the auto-researcher building blocks (oracle, runner helpers, ledger, report) into a real end-to-end orchestration loop, and implement the first three code-patch payloads so the tier 1 queue can actually execute.

**Architecture:** Two new modules: `decomposer/research/apply.py` (dispatch hypothesis kinds to file edits in the worktree) and `decomposer/research/run.py` (orchestration loop with stopping criteria). Plus three patch modules in `decomposer/research/patches/`. Wires into the existing `decomposer/research/cli.py`'s stubbed `run` and `replay` commands.

**Tech Stack:** Python 3.12, subprocess (running `uv run decomposer decompose` per experiment), all existing research building blocks.

**Spec:** `docs/superpowers/specs/2026-05-18-auto-researcher-design.md` ("stopping criteria", "failure handling", `apply.kind` semantics).

**v1 plan:** `docs/superpowers/plans/2026-05-18-auto-researcher.md` (foundations — landed).

---

## Pre-flight: verify worktree execution model

Before any task, verify that `cd <worktree> && uv run decomposer decompose ...` uses the worktree's `decomposer/core/` source (not the main repo's). Run this manually:

```bash
git worktree add worktrees/sanity -b sanity-check
cd worktrees/sanity
echo "# sanity tag" >> decomposer/core/mps_backend.py
uv run python -c "import decomposer.core.mps_backend; print(open(decomposer.core.mps_backend.__file__).read()[-30:])"
```

The output should include `# sanity tag`. If it does, the worktree model works as the v1 spec assumed. If it doesn't, every patch needs `uv pip install -e .` per worktree — flag as BLOCKED before writing patches.

Clean up: `cd .. && git worktree remove worktrees/sanity --force && git branch -D sanity-check`.

---

## File Map

```
decomposer/
└── research/
    ├── apply.py                      # NEW — dispatch hypothesis kinds to worktree edits
    ├── run.py                        # NEW — orchestration loop + stopping criteria
    ├── cli.py                        # MODIFY — wire `run` and `replay`
    └── patches/                      # exists
        ├── empty_cache_between_stages.py    # NEW
        ├── apply_first_block_cache.py        # NEW
        └── keep_warm_residency.py            # NEW

tests/research/
├── test_apply.py                     # NEW
├── test_run.py                       # NEW
├── test_patches.py                   # NEW
└── test_cli.py                       # MODIFY — add run/replay tests
```

---

## Task 1: `ExperimentResult` + `apply.py` skeleton

**Files:**
- Create: `decomposer/research/apply.py`
- Create: `tests/research/test_apply.py`

- [ ] **Step 1: Write failing test**

`tests/research/test_apply.py`:
```python
from pathlib import Path

import pytest

from decomposer.research.apply import ExperimentResult, apply_hypothesis
from decomposer.research.experiments import Apply, Hypothesis, HypothesisKind


def test_apply_hypothesis_env_var_is_noop(tmp_path):
    h = Hypothesis(
        id="x", description="",
        apply=Apply(kind=HypothesisKind.ENV_VAR, params={"FOO": "bar"}),
    )
    apply_hypothesis(h, tmp_path)


def test_apply_hypothesis_setting_change_is_noop(tmp_path):
    h = Hypothesis(
        id="x", description="",
        apply=Apply(kind=HypothesisKind.SETTING_CHANGE, params={"hf_repo": "x/y"}),
    )
    apply_hypothesis(h, tmp_path)


def test_experiment_result_has_required_fields():
    r = ExperimentResult(
        experiment_id="x",
        worktree_path=Path("/tmp/wt"),
        trace={},
        layers=[],
        exit_code=0,
        stderr="",
    )
    assert r.experiment_id == "x"
    assert r.exit_code == 0
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_apply.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `decomposer/research/apply.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from decomposer.research.experiments import Hypothesis, HypothesisKind


@dataclass
class ExperimentResult:
    experiment_id: str
    worktree_path: Path
    trace: dict[str, Any]
    layers: list[Image.Image]
    exit_code: int
    stderr: str
    duration_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def apply_hypothesis(hypothesis: Hypothesis, worktree_path: Path) -> None:
    kind = hypothesis.apply.kind
    if kind in (HypothesisKind.ENV_VAR, HypothesisKind.SETTING_CHANGE):
        return
    if kind == HypothesisKind.LORA_LOAD:
        _apply_lora_load(hypothesis, worktree_path)
        return
    if kind == HypothesisKind.SCHEDULER_SWAP:
        _apply_scheduler_swap(hypothesis, worktree_path)
        return
    if kind == HypothesisKind.CODE_PATCH:
        _apply_code_patch(hypothesis, worktree_path)
        return
    raise ValueError(f"unknown hypothesis kind: {kind}")


def _apply_lora_load(hypothesis: Hypothesis, worktree_path: Path) -> None:
    raise NotImplementedError("Task 2 implements this")


def _apply_scheduler_swap(hypothesis: Hypothesis, worktree_path: Path) -> None:
    raise NotImplementedError("Task 3 implements this")


def _apply_code_patch(hypothesis: Hypothesis, worktree_path: Path) -> None:
    raise NotImplementedError("Task 4 implements this")
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_apply.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/apply.py tests/research/test_apply.py
git commit -m "apply: ExperimentResult dataclass + apply_hypothesis dispatch skeleton"
```

---

## Task 2: `_apply_lora_load` implementation

**Files:**
- Modify: `decomposer/research/apply.py`
- Modify: `tests/research/test_apply.py`

The LoRA load hypothesis modifies `decomposer/core/mps_backend.py` inside the worktree to call `pipe.load_lora_weights(...)` right after the pipeline is built. Concretely: find the line `pipe = self._build_pipeline(dit, vae=None)` (or whatever the current form is) and inject a LoRA load + fuse call right after.

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_apply.py`:
```python
import subprocess


@pytest.fixture
def fake_worktree(tmp_path):
    """A minimal worktree with a fake mps_backend.py containing a known anchor line."""
    wt = tmp_path / "wt"
    wt.mkdir()
    (wt / "decomposer").mkdir()
    (wt / "decomposer" / "core").mkdir()
    (wt / "decomposer" / "core" / "mps_backend.py").write_text(
        "class MpsBackend:\n"
        "    def _denoise(self, ...):\n"
        "        pipe = self._build_pipeline(dit, vae=None)\n"
        "        # rest of denoise\n"
    )
    return wt


def test_apply_lora_load_injects_lora_call(fake_worktree):
    h = Hypothesis(
        id="lightning", description="",
        apply=Apply(kind=HypothesisKind.LORA_LOAD, params={
            "repo": "lightx2v/Qwen-Image-Lightning",
            "filename": "Lightning-8steps.safetensors",
            "scale": 1.0,
        }),
    )
    apply_hypothesis(h, fake_worktree)
    content = (fake_worktree / "decomposer" / "core" / "mps_backend.py").read_text()
    assert "load_lora_weights" in content
    assert "lightx2v/Qwen-Image-Lightning" in content
    assert "Lightning-8steps.safetensors" in content


def test_apply_lora_load_fails_when_anchor_missing(tmp_path):
    wt = tmp_path / "no-anchor"
    wt.mkdir()
    (wt / "decomposer").mkdir()
    (wt / "decomposer" / "core").mkdir()
    (wt / "decomposer" / "core" / "mps_backend.py").write_text("# no anchor here\n")
    h = Hypothesis(
        id="lightning", description="",
        apply=Apply(kind=HypothesisKind.LORA_LOAD, params={
            "repo": "r", "filename": "f", "scale": 1.0,
        }),
    )
    with pytest.raises(RuntimeError, match="anchor line not found"):
        apply_hypothesis(h, wt)
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_apply.py::test_apply_lora_load_injects_lora_call -v` → NotImplementedError or AssertionError.

- [ ] **Step 3: Implement `_apply_lora_load`**

Replace the `NotImplementedError` in `_apply_lora_load` with:
```python
def _apply_lora_load(hypothesis: Hypothesis, worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    content = backend_file.read_text()
    anchor = "pipe = self._build_pipeline(dit, vae=None)"
    if anchor not in content:
        raise RuntimeError(f"anchor line not found in {backend_file}: {anchor!r}")
    params = hypothesis.apply.params
    repo = params["repo"]
    filename = params["filename"]
    scale = float(params.get("scale", 1.0))
    injection = (
        f"        pipe.load_lora_weights({repo!r}, weight_name={filename!r})\n"
        f"        pipe.fuse_lora(lora_scale={scale})\n"
    )
    new_content = content.replace(anchor + "\n", anchor + "\n" + injection)
    backend_file.write_text(new_content)
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_apply.py -v` → 5 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/apply.py tests/research/test_apply.py
git commit -m "apply: implement _apply_lora_load via mps_backend.py injection"
```

---

## Task 3: `_apply_scheduler_swap` implementation

**Files:**
- Modify: `decomposer/research/apply.py`
- Modify: `tests/research/test_apply.py`

The hypothesis params include `scheduler_class: UniPCMultistepScheduler` (a name string). The patch finds the existing `FlowMatchEulerDiscreteScheduler` import and instantiation in `mps_backend.py` (inside `_build_pipeline`) and swaps it.

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_apply.py`:
```python
@pytest.fixture
def fake_worktree_with_scheduler(tmp_path):
    wt = tmp_path / "wt-sched"
    wt.mkdir()
    (wt / "decomposer").mkdir()
    (wt / "decomposer" / "core").mkdir()
    (wt / "decomposer" / "core" / "mps_backend.py").write_text(
        "from diffusers import FlowMatchEulerDiscreteScheduler\n"
        "class MpsBackend:\n"
        "    def _build_pipeline(self, dit, vae):\n"
        "        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(\n"
        "            self.settings.hf_repo, subfolder='scheduler'\n"
        "        )\n"
    )
    return wt


def test_apply_scheduler_swap_replaces_class(fake_worktree_with_scheduler):
    h = Hypothesis(
        id="unipc", description="",
        apply=Apply(kind=HypothesisKind.SCHEDULER_SWAP, params={
            "scheduler_class": "UniPCMultistepScheduler",
        }),
    )
    apply_hypothesis(h, fake_worktree_with_scheduler)
    content = (fake_worktree_with_scheduler / "decomposer" / "core" / "mps_backend.py").read_text()
    assert "UniPCMultistepScheduler" in content
    assert "FlowMatchEulerDiscreteScheduler" not in content
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_apply.py::test_apply_scheduler_swap_replaces_class -v` → NotImplementedError.

- [ ] **Step 3: Implement `_apply_scheduler_swap`**

Replace the `NotImplementedError` in `_apply_scheduler_swap` with:
```python
def _apply_scheduler_swap(hypothesis: Hypothesis, worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    new_class = hypothesis.apply.params["scheduler_class"]
    content = backend_file.read_text()
    old_class = "FlowMatchEulerDiscreteScheduler"
    if old_class not in content:
        raise RuntimeError(f"scheduler class {old_class!r} not found in {backend_file}")
    content = content.replace(old_class, new_class)
    backend_file.write_text(content)
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_apply.py -v` → 6 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/apply.py tests/research/test_apply.py
git commit -m "apply: implement _apply_scheduler_swap"
```

---

## Task 4: `_apply_code_patch` dispatcher (imports + runs patch modules)

**Files:**
- Modify: `decomposer/research/apply.py`
- Modify: `tests/research/test_apply.py`

Each patch is a Python module in `decomposer/research/patches/<name>.py` exporting `apply(worktree_path: Path) -> None`. The dispatcher loads each named module and calls its `apply()`.

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_apply.py`:
```python
def test_apply_code_patch_runs_named_patch(tmp_path, monkeypatch):
    # Create a fake patch module that writes a sentinel file
    import sys
    import types
    fake = types.ModuleType("decomposer.research.patches.fake_test_patch")
    def fake_apply(wt_path):
        (wt_path / "PATCHED").write_text("ok")
    fake.apply = fake_apply
    monkeypatch.setitem(sys.modules, "decomposer.research.patches.fake_test_patch", fake)

    h = Hypothesis(
        id="x", description="",
        apply=Apply(kind=HypothesisKind.CODE_PATCH, params={
            "patches": ["fake_test_patch"],
        }),
    )
    apply_hypothesis(h, tmp_path)
    assert (tmp_path / "PATCHED").exists()


def test_apply_code_patch_fails_on_missing_patch(tmp_path):
    h = Hypothesis(
        id="x", description="",
        apply=Apply(kind=HypothesisKind.CODE_PATCH, params={
            "patches": ["nonexistent_patch_xyz"],
        }),
    )
    with pytest.raises(ModuleNotFoundError):
        apply_hypothesis(h, tmp_path)
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_apply.py::test_apply_code_patch_runs_named_patch -v` → NotImplementedError.

- [ ] **Step 3: Implement `_apply_code_patch`**

Replace the `NotImplementedError` in `_apply_code_patch` with:
```python
def _apply_code_patch(hypothesis: Hypothesis, worktree_path: Path) -> None:
    import importlib

    patch_names = hypothesis.apply.params.get("patches", [])
    if not patch_names:
        raise ValueError(f"code_patch hypothesis {hypothesis.id!r} has empty 'patches' list")
    for name in patch_names:
        module = importlib.import_module(f"decomposer.research.patches.{name}")
        if not hasattr(module, "apply"):
            raise RuntimeError(
                f"patch module {name!r} does not export an apply(worktree_path) function"
            )
        module.apply(worktree_path)
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_apply.py -v` → 8 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/apply.py tests/research/test_apply.py
git commit -m "apply: code_patch dispatcher imports and runs named patches"
```

---

## Task 5: First patch — `empty_cache_between_stages`

**Files:**
- Create: `decomposer/research/patches/empty_cache_between_stages.py`
- Create: `tests/research/test_patches.py`

Inserts `torch.mps.empty_cache()` calls into `mps_backend.py` between major stages (after `free_text_encoder`, after `free_dit`, etc.) to test if more aggressive cache flushing helps.

- [ ] **Step 1: Write failing test**

Create `tests/research/test_patches.py`:
```python
from pathlib import Path

import pytest


@pytest.fixture
def worktree_mps_backend(tmp_path):
    wt = tmp_path / "wt"
    (wt / "decomposer" / "core").mkdir(parents=True)
    (wt / "decomposer" / "core" / "mps_backend.py").write_text(
        "import torch\n"
        "import gc\n"
        "\n"
        "class MpsBackend:\n"
        "    def _encode_prompt(self, ...):\n"
        "        with tracer.stage('free_text_encoder'):\n"
        "            del te\n"
        "            self.residency.free()\n"
        "            gc.collect()\n"
        "\n"
        "    def _denoise(self, ...):\n"
        "        with tracer.stage('free_dit'):\n"
        "            del pipe, dit\n"
        "            self.residency.free()\n"
        "            gc.collect()\n"
    )
    return wt


def test_empty_cache_patch_inserts_calls(worktree_mps_backend):
    from decomposer.research.patches import empty_cache_between_stages
    empty_cache_between_stages.apply(worktree_mps_backend)
    content = (worktree_mps_backend / "decomposer" / "core" / "mps_backend.py").read_text()
    assert content.count("torch.mps.empty_cache()") >= 2
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_patches.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `decomposer/research/patches/empty_cache_between_stages.py`**

```python
from pathlib import Path


def apply(worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    content = backend_file.read_text()
    new_content = content.replace(
        "self.residency.free()\n            gc.collect()",
        "self.residency.free()\n            gc.collect()\n            "
        "torch.mps.empty_cache() if torch.backends.mps.is_available() else None",
    )
    if new_content == content:
        raise RuntimeError(f"no residency.free()+gc.collect() pattern found in {backend_file}")
    backend_file.write_text(new_content)
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_patches.py -v` → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/patches/empty_cache_between_stages.py tests/research/test_patches.py
git commit -m "patches: empty_cache_between_stages — insert torch.mps.empty_cache() after each free_* stage"
```

---

## Task 6: Second patch — `apply_first_block_cache`

**Files:**
- Create: `decomposer/research/patches/apply_first_block_cache.py`
- Modify: `tests/research/test_patches.py`

Wraps `pipe.transformer` in a `diffusers.hooks.apply_first_block_cache(...)` call right after the pipeline is built, with threshold 0.08.

- [ ] **Step 1: Append failing test**

Append to `tests/research/test_patches.py`:
```python
def test_first_block_cache_patch_inserts_apply_call(worktree_mps_backend):
    # Add an anchor line the patch can find
    file = worktree_mps_backend / "decomposer" / "core" / "mps_backend.py"
    file.write_text(file.read_text() +
        "    def _denoise2(self, ...):\n"
        "        pipe = self._build_pipeline(dit, vae=None)\n"
        "        # rest\n"
    )
    from decomposer.research.patches import apply_first_block_cache
    apply_first_block_cache.apply(worktree_mps_backend)
    content = file.read_text()
    assert "apply_first_block_cache" in content
    assert "FirstBlockCacheConfig" in content
    assert "0.08" in content
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_patches.py::test_first_block_cache_patch_inserts_apply_call -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `decomposer/research/patches/apply_first_block_cache.py`**

```python
from pathlib import Path


THRESHOLD = 0.08
ANCHOR = "pipe = self._build_pipeline(dit, vae=None)"


def apply(worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    content = backend_file.read_text()
    if ANCHOR not in content:
        raise RuntimeError(f"anchor line not found in {backend_file}: {ANCHOR!r}")
    injection = (
        f"        from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig\n"
        f"        apply_first_block_cache(pipe.transformer, FirstBlockCacheConfig(threshold={THRESHOLD}))\n"
    )
    new_content = content.replace(ANCHOR + "\n", ANCHOR + "\n" + injection)
    backend_file.write_text(new_content)
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_patches.py -v` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/patches/apply_first_block_cache.py tests/research/test_patches.py
git commit -m "patches: apply_first_block_cache at threshold 0.08"
```

---

## Task 7: Third patch — `keep_warm_residency`

**Files:**
- Create: `decomposer/research/patches/keep_warm_residency.py`
- Modify: `tests/research/test_patches.py`

Replaces `ResidencyManager.free()` no-ops where it makes sense — specifically, the patch comments out the `self.residency.free()` calls in `mps_backend.py` so models stay resident across requests. Combined with smaller Q4 weights this enables keep-warm.

The simplest patch: comment out every `self.residency.free()` line.

- [ ] **Step 1: Append failing test**

Append to `tests/research/test_patches.py`:
```python
def test_keep_warm_patch_disables_residency_free(worktree_mps_backend):
    from decomposer.research.patches import keep_warm_residency
    keep_warm_residency.apply(worktree_mps_backend)
    content = (worktree_mps_backend / "decomposer" / "core" / "mps_backend.py").read_text()
    # No active self.residency.free() calls remain
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("self.residency.free()"):
            pytest.fail(f"residency.free() still active: {line!r}")
        # but a commented version should exist
    assert "# keep-warm: self.residency.free()" in content
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_patches.py::test_keep_warm_patch_disables_residency_free -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `decomposer/research/patches/keep_warm_residency.py`**

```python
from pathlib import Path


def apply(worktree_path: Path) -> None:
    backend_file = worktree_path / "decomposer" / "core" / "mps_backend.py"
    if not backend_file.exists():
        raise RuntimeError(f"mps_backend.py not found at {backend_file}")
    content = backend_file.read_text()
    out_lines: list[str] = []
    replaced = 0
    for line in content.splitlines():
        if line.lstrip().startswith("self.residency.free()"):
            indent = line[: len(line) - len(line.lstrip())]
            out_lines.append(f"{indent}# keep-warm: self.residency.free()")
            replaced += 1
        else:
            out_lines.append(line)
    if replaced == 0:
        raise RuntimeError(
            f"no self.residency.free() lines found in {backend_file} — nothing to disable"
        )
    backend_file.write_text("\n".join(out_lines) + ("\n" if content.endswith("\n") else ""))
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_patches.py -v` → 3 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/patches/keep_warm_residency.py tests/research/test_patches.py
git commit -m "patches: keep_warm_residency — disable residency.free() calls"
```

---

## Task 8: `run.py` — single-experiment orchestration

**Files:**
- Create: `decomposer/research/run.py`
- Create: `tests/research/test_run.py`

`run_experiment(hypothesis, baseline, repo, ...)` ties together: create_worktree → apply_hypothesis → run subprocess → parse trace & layers → score → decide → archive_or_promote → ledger.append.

- [ ] **Step 1: Write failing test**

Create `tests/research/test_run.py`:
```python
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from decomposer.research.apply import ExperimentResult
from decomposer.research.baseline import Baseline
from decomposer.research.experiments import Apply, Hypothesis, HypothesisKind
from decomposer.research.run import run_experiment


@pytest.fixture
def fake_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "decomposer").mkdir()
    (repo / "decomposer" / "core").mkdir()
    (repo / "decomposer" / "core" / "mps_backend.py").write_text("# placeholder\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    return repo


@pytest.fixture
def baseline_obj(tmp_path):
    image = Image.new("RGB", (32, 32), (100, 100, 100))
    layers = [Image.new("RGBA", (32, 32), (50, 0, 0, 200))]
    trace = {"run_id": "bl", "total_wall_ms": 1000.0,
             "stages": [{"name": "load_dit", "wall_ms": 500.0}]}
    return Baseline(run_id="bl", input_image=image, layers=layers, trace=trace,
                    layers_count=1, resolution=640, steps=8, commit_sha="abc")


def _fake_run_decompose(cmd, env, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    layer = Image.new("RGBA", (32, 32), (50, 0, 0, 200))
    layer.save(out / "layer_0.png")
    trace = {"run_id": "exp", "total_wall_ms": 500.0,
             "stages": [{"name": "load_dit", "wall_ms": 250.0}]}
    (out / "trace.json").write_text(json.dumps(trace))
    return ExperimentResult(experiment_id="x", worktree_path=Path("/tmp"),
                             trace=trace, layers=[layer], exit_code=0, stderr="")


def test_run_experiment_invokes_apply_and_subprocess(fake_repo, baseline_obj, tmp_path):
    h = Hypothesis(
        id="env-test", description="",
        apply=Apply(kind=HypothesisKind.ENV_VAR, params={}),
        quality_bounds={"composite_ssim_min": 0.5, "per_layer_ssim_min": 0.5},
    )
    ledger_path = tmp_path / "ledger.jsonl"
    with patch("decomposer.research.run._run_decompose", side_effect=_fake_run_decompose):
        decision, entry = run_experiment(
            h, baseline_obj, fake_repo, ledger_path=ledger_path,
            image_path=str(tmp_path / "img.jpg"),
        )
    assert entry.experiment_id == "env-test"
    assert ledger_path.exists()
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_run.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `decomposer/research/run.py`**

```python
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from decomposer.research.apply import ExperimentResult, apply_hypothesis
from decomposer.research.baseline import Baseline
from decomposer.research.experiments import Hypothesis
from decomposer.research.ledger import LedgerEntry, append as append_ledger
from decomposer.research.oracle import score as score_quality
from decomposer.research.runner import (
    archive_worktree,
    build_decompose_args,
    compare_traces,
    create_worktree,
    decide,
    promote_worktree,
    Decision,
)


def _run_decompose(cmd: list[str], env: dict[str, str], out_dir: str) -> ExperimentResult:
    full_env = os.environ.copy()
    full_env.update(env)
    start = time.perf_counter()
    proc = subprocess.run(cmd, env=full_env, capture_output=True, text=True, timeout=1800)
    duration = time.perf_counter() - start
    out_path = Path(out_dir)
    trace_path = out_path / "trace.json"
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else {}
    layers: list[Image.Image] = []
    i = 0
    while True:
        layer_path = out_path / f"layer_{i}.png"
        if not layer_path.exists():
            break
        layers.append(Image.open(layer_path).convert("RGBA"))
        i += 1
    return ExperimentResult(
        experiment_id="",
        worktree_path=out_path.parent,
        trace=trace,
        layers=layers,
        exit_code=proc.returncode,
        stderr=proc.stderr,
        duration_seconds=duration,
    )


def run_experiment(
    hypothesis: Hypothesis,
    baseline: Baseline,
    repo: Path,
    *,
    ledger_path: Path,
    image_path: str,
    base_layers: int = 3,
    base_resolution: int = 640,
    base_steps: int = 8,
) -> tuple[Decision, LedgerEntry]:
    worktree = create_worktree(repo, hypothesis.id)
    apply_hypothesis(hypothesis, worktree)

    out_dir = str(worktree / "run-out")
    cmd, env = build_decompose_args(
        hypothesis, image_path=image_path,
        base_steps=base_steps, base_resolution=base_resolution,
        base_layers=base_layers, out_dir=out_dir,
    )
    result = _run_decompose(cmd, env, out_dir)
    result.experiment_id = hypothesis.id

    quality = score_quality(result.layers, baseline.layers, baseline.input_image)
    perf = compare_traces(baseline.trace, result.trace)
    decision = decide(quality, perf, hypothesis.quality_bounds)

    merged_sha = None
    if decision == Decision.MERGE:
        merged_sha = promote_worktree(repo, hypothesis.id)
    else:
        archive_worktree(repo, hypothesis.id, decision.value)

    entry = LedgerEntry(
        timestamp=_dt.datetime.utcnow().isoformat() + "Z",
        experiment_id=hypothesis.id,
        baseline_run_id=baseline.run_id,
        experiment_run_id=result.trace.get("run_id", "unknown"),
        decision=decision.value,
        perf={"baseline_total_wall_ms": perf.baseline_total_wall_ms,
              "experiment_total_wall_ms": perf.experiment_total_wall_ms,
              "delta_pct": perf.delta_pct,
              "stage_deltas": perf.stage_deltas},
        quality={"composite_ssim": quality.composite_ssim,
                 "per_layer_ssim_matched": quality.per_layer_ssim_matched,
                 "non_degenerate": quality.non_degenerate,
                 "degeneracy_reasons": quality.degeneracy_reasons,
                 "notes": quality.notes},
        hypothesis_summary=hypothesis.description,
        merged_commit_sha=merged_sha,
        worktree_path=str(worktree),
        human_audit_pending=(decision == Decision.MERGE),
        rejection_reason=None if decision == Decision.MERGE else decision.value,
    )
    append_ledger(ledger_path, entry)
    return decision, entry
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_run.py -v` → 1 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/run.py tests/research/test_run.py
git commit -m "run: run_experiment ties apply+subprocess+score+decide+ledger together"
```

---

## Task 9: `StopCondition` + `run_queue` loop

**Files:**
- Modify: `decomposer/research/run.py`
- Modify: `tests/research/test_run.py`

`run_queue(...)` iterates hypotheses, calling `run_experiment` for each. Stops when ANY of:
1. Wall-time budget exhausted
2. Target latency reached
3. Max experiments hit
4. Three consecutive non-MERGE
5. Queue exhausted

- [ ] **Step 1: Write failing test**

Append to `tests/research/test_run.py`:
```python
from decomposer.research.run import StopCondition, run_queue


def test_stop_condition_max_experiments_triggers():
    sc = StopCondition(max_experiments=2)
    assert sc.should_stop(experiments_run=2, consecutive_non_merges=0,
                          current_latency_ms=1000.0, wall_seconds=10.0)[0] is True


def test_stop_condition_consecutive_non_merges():
    sc = StopCondition()
    assert sc.should_stop(0, 0, 1000.0, 0.0)[0] is False
    assert sc.should_stop(3, 3, 1000.0, 0.0)[0] is True


def test_stop_condition_target_latency_reached():
    sc = StopCondition(target_latency_seconds=10.0)
    assert sc.should_stop(1, 0, 9000.0, 0.0)[0] is True
    assert sc.should_stop(1, 0, 11000.0, 0.0)[0] is False


def test_stop_condition_budget_exhausted():
    sc = StopCondition(budget_seconds=60.0)
    assert sc.should_stop(1, 0, 100.0, 61.0)[0] is True
    assert sc.should_stop(1, 0, 100.0, 59.0)[0] is False
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_run.py::test_stop_condition_max_experiments_triggers -v` → ImportError.

- [ ] **Step 3: Append to `decomposer/research/run.py`**

```python
from dataclasses import dataclass, field


@dataclass
class StopCondition:
    budget_seconds: float | None = None
    target_latency_seconds: float | None = None
    max_experiments: int | None = None
    max_consecutive_non_merges: int = 3

    def should_stop(
        self,
        experiments_run: int,
        consecutive_non_merges: int,
        current_latency_ms: float,
        wall_seconds: float,
    ) -> tuple[bool, str]:
        if self.max_experiments is not None and experiments_run >= self.max_experiments:
            return True, "max_experiments"
        if self.budget_seconds is not None and wall_seconds >= self.budget_seconds:
            return True, "budget"
        if self.target_latency_seconds is not None and current_latency_ms <= self.target_latency_seconds * 1000.0:
            return True, "target_latency"
        if consecutive_non_merges >= self.max_consecutive_non_merges:
            return True, "consecutive_non_merges"
        return False, ""


def run_queue(
    queue: list[Hypothesis],
    baseline: Baseline,
    repo: Path,
    *,
    ledger_path: Path,
    image_path: str,
    stop: StopCondition,
    base_layers: int = 3,
    base_resolution: int = 640,
    base_steps: int = 8,
) -> list[LedgerEntry]:
    entries: list[LedgerEntry] = []
    consecutive_non_merges = 0
    current_latency_ms = baseline.trace.get("total_wall_ms", float("inf"))
    started = time.perf_counter()
    for hypothesis in queue:
        wall = time.perf_counter() - started
        stop_now, reason = stop.should_stop(
            experiments_run=len(entries),
            consecutive_non_merges=consecutive_non_merges,
            current_latency_ms=current_latency_ms,
            wall_seconds=wall,
        )
        if stop_now:
            break
        decision, entry = run_experiment(
            hypothesis, baseline, repo,
            ledger_path=ledger_path, image_path=image_path,
            base_layers=base_layers, base_resolution=base_resolution, base_steps=base_steps,
        )
        entries.append(entry)
        if decision == Decision.MERGE:
            consecutive_non_merges = 0
            current_latency_ms = entry.perf["experiment_total_wall_ms"]
        else:
            consecutive_non_merges += 1
    return entries
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_run.py -v` → 5 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/research/run.py tests/research/test_run.py
git commit -m "run: StopCondition + run_queue loop with 5 stopping criteria"
```

---

## Task 10: Wire `cli.run` and `cli.replay` to call the orchestrator

**Files:**
- Modify: `decomposer/research/cli.py`
- Modify: `tests/research/test_cli.py`

- [ ] **Step 1: Append failing test**

Append to `tests/research/test_cli.py`:
```python
def test_research_run_loads_queue_and_invokes_loop(tmp_path, monkeypatch):
    """Confirms cli.run parses --queue + flag args and calls run_queue. Mock the loop."""
    queue_file = tmp_path / "q.yaml"
    queue_file.write_text("""
experiments:
  - id: x
    description: ""
    apply:
      kind: env_var
      params: {}
    quality_bounds:
      composite_ssim_min: 0.5
      per_layer_ssim_min: 0.5
""")
    called = {}

    def fake_run_queue(*args, **kwargs):
        called["yes"] = True
        called["stop"] = kwargs.get("stop")
        return []

    monkeypatch.setenv("DECOMPOSER_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr("decomposer.research.cli.run_queue", fake_run_queue)
    monkeypatch.setattr("decomposer.research.cli._load_baseline_or_die",
                        lambda *_a, **_k: None)
    monkeypatch.setattr("decomposer.research.cli._repo_root", lambda: tmp_path)

    result = runner.invoke(app, ["research", "run",
                                  "--queue", str(queue_file),
                                  "--budget", "1h",
                                  "--target-latency", "30s",
                                  "--max-experiments", "5"])
    assert called.get("yes") is True
    assert called["stop"].max_experiments == 5
    assert result.exit_code == 0
```

- [ ] **Step 2: Confirm fail**

`uv run pytest tests/research/test_cli.py::test_research_run_loads_queue_and_invokes_loop -v` → AttributeError or assertion failure.

- [ ] **Step 3: Modify `decomposer/research/cli.py`**

Replace the existing stubbed `run` command with this; also add helpers `_parse_duration`, `_load_baseline_or_die`, `_repo_root`:

```python
import re
from decomposer.research.baseline import load_baseline
from decomposer.research.experiments import load_queue
from decomposer.research.run import StopCondition, run_queue


def _parse_duration(text: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", text.strip().lower())
    if not match:
        raise typer.BadParameter(f"duration must look like '30s', '5m', '8h': got {text!r}")
    value = float(match.group(1))
    return value * {"s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]


def _repo_root() -> Path:
    return Path.cwd()


def _load_baseline_or_die(settings) -> "Baseline":
    bl_dir = settings.runs_dir / "baseline-latest"
    if not bl_dir.exists():
        console.print(f"[red]No baseline at {bl_dir}. Run `decomposer research baseline` first.[/red]")
        raise typer.Exit(1)
    return load_baseline(bl_dir)


@research_app.command()
def run(
    queue: Annotated[Path, typer.Option(exists=True, readable=True)],
    budget: str = "8h",
    target_latency: str = "30s",
    max_experiments: int = 10,
) -> None:
    """Execute experiments from the queue (sequential)."""
    settings = get_settings()
    baseline = _load_baseline_or_die(settings)
    hypotheses = load_queue(queue)
    stop = StopCondition(
        budget_seconds=_parse_duration(budget),
        target_latency_seconds=_parse_duration(target_latency),
        max_experiments=max_experiments,
    )
    ledger_path = settings.runs_dir / "ledger.jsonl"
    entries = run_queue(
        hypotheses, baseline, _repo_root(),
        ledger_path=ledger_path,
        image_path=str(Path("test_image.jpg").resolve()),
        stop=stop,
    )
    console.print(f"[bold]run complete[/bold] {len(entries)} experiments processed")
    merged = sum(1 for e in entries if e.decision == "MERGE")
    console.print(f"  {merged} merged, {len(entries) - merged} not merged")
```

And replace the `replay` stub with:
```python
@research_app.command()
def replay(experiment_id: str) -> None:
    """Re-run a specific experiment from the ledger."""
    settings = get_settings()
    ledger_path = settings.runs_dir / "ledger.jsonl"
    entries = read_all(ledger_path)
    matching = [e for e in entries if e.experiment_id == experiment_id]
    if not matching:
        console.print(f"[red]no experiment {experiment_id!r} in ledger[/red]")
        raise typer.Exit(1)
    entry = matching[-1]
    console.print(f"Most recent run for {experiment_id!r}:")
    console.print(f"  decision: {entry.decision}")
    console.print(f"  perf delta_pct: {entry.perf.get('delta_pct'):.2f}%")
    console.print(f"  quality: composite={entry.quality.get('composite_ssim'):.3f} "
                  f"per_layer={entry.quality.get('per_layer_ssim_matched'):.3f}")
    console.print("[yellow]Re-execution is not yet wired; this prints the historical result.[/yellow]")
```

- [ ] **Step 4: Confirm pass**

`uv run pytest tests/research/test_cli.py -v` → 3 passed.

- [ ] **Step 5: Manual smoke**

```bash
uv run decomposer research run --help
```
Expected: usage with --queue, --budget, --target-latency, --max-experiments.

```bash
uv run decomposer research replay nonexistent-id
```
Expected: exit 1, "no experiment 'nonexistent-id' in ledger".

- [ ] **Step 6: Commit**

```bash
git add decomposer/research/cli.py tests/research/test_cli.py
git commit -m "cli: wire research run (queue + StopCondition) and replay"
```

---

## Done criteria

- All 10 tasks complete with TDD
- Full non-MPS suite green (`uv run pytest -m "not mps_required"`) — expect ~120 tests
- `decomposer research run --queue docs/superpowers/research/queue.yaml --max-experiments 1 --target-latency 1s` is now invokable (will need a baseline + test_image.jpg + can fail at first real subprocess call, but the orchestration plumbing is in place)
- 3 code-patch payloads usable from the queue: `empty_cache_between_stages`, `apply_first_block_cache`, `keep_warm_residency`
- Apply layer handles LORA_LOAD, SCHEDULER_SWAP, CODE_PATCH, ENV_VAR, SETTING_CHANGE
- `decomposer research replay <id>` shows historical experiment record

## Out of scope (for a future v3 plan)

- The remaining 4 code-patch payloads referenced in `queue.yaml`: `port_q5km_dequant`, `swap_gguf_file_to_q5km`, `port_q4km_dequant`, `swap_gguf_file_to_q4km`, `apply_torch_compile_to_transformer`, `pre_allocate_denoise_latent`. Each requires research-level work (porting k-quant dequant math from city96, dispatcher extension to GgufLinear, etc.).
- VLM spot-check audit queue + `decomposer research audit` command
- `decomposer research revert <id>` for rolling back a merged experiment
- Parallel experiment execution across multiple worktrees
- The MLX port (v3 spec)

## Open risks

| Risk | Mitigation |
|---|---|
| `cd worktree && uv run decomposer ...` doesn't pick up the worktree's edited source | Pre-flight verification (Pre-flight section above) — verify before Task 1 |
| First Lightning LoRA test passes on quality but produces grid artifacts SSIM can't detect | Manual VLM audit on first MERGE before running the rest of the queue |
| `apply_first_block_cache` import fails on pinned diffusers commit | Catch in `_apply_code_patch` and surface clearly; flag for diffusers re-pin if needed |
| Worktree experiments pollute `runs/` of the main repo with `runs/exp-<id>/` outputs | Use `worktree/run-out/` inside the worktree itself for outputs |
| Subprocess timeout (1800s) trips for a real ~12-minute Mac run | Configurable via Settings.inference_timeout_seconds — already exists |
