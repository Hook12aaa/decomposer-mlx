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


def _run_decompose(cmd: list[str], env: dict[str, str], out_dir: str,
                    cwd: str | Path | None = None) -> ExperimentResult:
    full_env = os.environ.copy()
    full_env.update(env)
    start = time.perf_counter()
    proc = subprocess.run(cmd, env=full_env, capture_output=True, text=True,
                          timeout=1800, cwd=cwd)
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


def _commit_worktree_changes(worktree: Path, hypothesis_id: str) -> None:
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=worktree, capture_output=True, text=True
    )
    if result.stdout.strip():
        subprocess.run(["git", "add", "-A"], cwd=worktree, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", f"apply hypothesis: {hypothesis_id}"],
            cwd=worktree, check=True,
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
    _commit_worktree_changes(worktree, hypothesis.id)

    out_dir = str(worktree / "run-out")
    cmd, env = build_decompose_args(
        hypothesis, image_path=image_path,
        base_steps=base_steps, base_resolution=base_resolution,
        base_layers=base_layers, out_dir=out_dir,
    )
    result = _run_decompose(cmd, env, out_dir, cwd=worktree)
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
                 "max_alpha_coverage_diff": quality.max_alpha_coverage_diff,
                 "max_rgb_mean_distance": quality.max_rgb_mean_distance,
                 "degeneracy_reasons": quality.degeneracy_reasons,
                 "notes": quality.notes},
        hypothesis_summary=hypothesis.description,
        merged_commit_sha=merged_sha,
        worktree_path=str(worktree),
        human_audit_pending=(decision == Decision.MERGE),
        rejection_reason=None if decision == Decision.MERGE else decision.value,
        extras={"stderr": result.stderr[-4000:] if result.stderr else "",
                "exit_code": result.exit_code,
                "duration_seconds": result.duration_seconds},
    )
    append_ledger(ledger_path, entry)
    return decision, entry


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
