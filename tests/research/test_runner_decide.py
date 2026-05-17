from decomposer.research.oracle import QualityReport
from decomposer.research.runner import Decision, PerfReport, decide


def _qr(composite=0.95, matched=0.90, non_degen=True, alpha_diff=0.02, rgb_dist=3.0):
    return QualityReport(
        composite_ssim=composite,
        per_layer_ssim_matched=matched,
        per_layer_ssim_individual=[matched] * 3,
        layer_match_indices=[0, 1, 2],
        non_degenerate=non_degen,
        max_alpha_coverage_diff=alpha_diff,
        max_rgb_mean_distance=rgb_dist,
    )


def _perf(delta_pct):
    return PerfReport(
        baseline_total_wall_ms=100000.0,
        experiment_total_wall_ms=100000.0 * (1 + delta_pct / 100.0),
        delta_pct=delta_pct,
        stage_deltas={},
    )


def test_merges_when_faster_and_quality_passes():
    bounds = {"composite_ssim_min": 0.60, "per_layer_ssim_min": 0.50}
    assert decide(_qr(), _perf(-10.0), bounds) == Decision.MERGE


def test_rejects_degenerate_layers():
    bounds = {}
    assert decide(_qr(non_degen=False), _perf(-10.0), bounds) == Decision.REJECT_DEGENERATE


def test_rejects_quality_loss_ssim():
    bounds = {"composite_ssim_min": 0.60, "per_layer_ssim_min": 0.50}
    assert decide(_qr(composite=0.40), _perf(-10.0), bounds) == Decision.REJECT_QUALITY
    assert decide(_qr(matched=0.30), _perf(-10.0), bounds) == Decision.REJECT_QUALITY


def test_rejects_quality_loss_structural():
    bounds = {"max_alpha_diff": 0.15, "max_rgb_dist": 20.0}
    assert decide(_qr(alpha_diff=0.25), _perf(-10.0), bounds) == Decision.REJECT_QUALITY
    assert decide(_qr(rgb_dist=30.0), _perf(-10.0), bounds) == Decision.REJECT_QUALITY


def test_rejects_significant_regression():
    bounds = {}
    assert decide(_qr(), _perf(+6.0), bounds) == Decision.REJECT_REGRESSION


def test_keeps_for_review_when_near_neutral():
    bounds = {}
    assert decide(_qr(), _perf(-1.0), bounds) == Decision.KEEP_FOR_REVIEW
    assert decide(_qr(), _perf(+2.0), bounds) == Decision.KEEP_FOR_REVIEW


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
