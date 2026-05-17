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
    max_alpha = quality_bounds.get("max_alpha_diff", 0.15)
    max_rgb = quality_bounds.get("max_rgb_dist", 20.0)
    if quality.max_alpha_coverage_diff > max_alpha:
        return Decision.REJECT_QUALITY
    if quality.max_rgb_mean_distance > max_rgb:
        return Decision.REJECT_QUALITY
    composite_min = quality_bounds.get("composite_ssim_min", 0.60)
    per_layer_min = quality_bounds.get("per_layer_ssim_min", 0.50)
    if quality.composite_ssim < composite_min:
        return Decision.REJECT_QUALITY
    if quality.per_layer_ssim_matched < per_layer_min:
        return Decision.REJECT_QUALITY
    if perf.delta_pct > REJECT_REGRESSION_THRESHOLD_PCT:
        return Decision.REJECT_REGRESSION
    if perf.delta_pct < MERGE_FASTER_THRESHOLD_PCT:
        return Decision.MERGE
    return Decision.KEEP_FOR_REVIEW


def compare_traces(baseline_trace: dict, experiment_trace: dict) -> PerfReport:
    base_total = float(baseline_trace.get("total_wall_ms", 0.0))
    exp_total = float(experiment_trace.get("total_wall_ms", 0.0))
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
    wt_path = repo / WORKTREES_DIR / exp_id
    branch = f"research/{exp_id}"
    wt_path.parent.mkdir(parents=True, exist_ok=True)
    existing = subprocess.run(
        ["git", "branch", "--list", branch], cwd=repo, capture_output=True, text=True
    )
    if branch in existing.stdout:
        wt_list = subprocess.run(
            ["git", "worktree", "list", "--porcelain"], cwd=repo,
            capture_output=True, text=True,
        )
        for line in wt_list.stdout.split("\n"):
            if line.startswith("worktree "):
                current_path = line[len("worktree "):]
            if f"branch refs/heads/{branch}" in line and current_path:
                subprocess.run(
                    ["git", "worktree", "remove", current_path, "--force"],
                    cwd=repo, capture_output=True,
                )
        subprocess.run(["git", "worktree", "prune"], cwd=repo, capture_output=True)
        subprocess.run(
            ["git", "branch", "-D", branch], cwd=repo, capture_output=True,
        )
    _run_git(repo, "worktree", "add", str(wt_path), "-b", branch)
    return wt_path


def promote_worktree(repo: Path, exp_id: str) -> str:
    branch = f"research/{exp_id}"
    _run_git(repo, "merge", "--no-ff", branch, "-m", f"Merge {branch} (MERGE)")
    sha = _run_git(repo, "rev-parse", "HEAD")
    _run_git(repo, "worktree", "remove", str(repo / WORKTREES_DIR / exp_id), "--force")
    _run_git(repo, "branch", "-D", branch)
    return sha


def archive_worktree(repo: Path, exp_id: str, reason: str) -> Path:
    src = repo / WORKTREES_DIR / exp_id
    dst_parent = repo / ARCHIVE_DIR
    dst_parent.mkdir(parents=True, exist_ok=True)
    dst = dst_parent / f"{exp_id}-{reason}"
    _run_git(repo, "worktree", "move", str(src), str(dst))
    return dst
