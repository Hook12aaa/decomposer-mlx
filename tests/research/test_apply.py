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


@pytest.fixture
def fake_worktree(tmp_path):
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


def test_apply_code_patch_runs_named_patch(tmp_path, monkeypatch):
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
