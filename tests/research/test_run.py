import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from decomposer.research.apply import ExperimentResult
from decomposer.research.baseline import Baseline
from decomposer.research.experiments import Apply, Hypothesis, HypothesisKind
from decomposer.research.run import run_experiment, StopCondition, run_queue


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


def _fake_run_decompose(cmd, env, out_dir, cwd=None):
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
