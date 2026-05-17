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
