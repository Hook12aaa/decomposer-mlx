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


def test_first_block_cache_patch_inserts_apply_call(worktree_mps_backend):
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


def test_keep_warm_patch_disables_residency_free(worktree_mps_backend):
    from decomposer.research.patches import keep_warm_residency
    keep_warm_residency.apply(worktree_mps_backend)
    content = (worktree_mps_backend / "decomposer" / "core" / "mps_backend.py").read_text()
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("self.residency.free()"):
            pytest.fail(f"residency.free() still active: {line!r}")
    assert "# keep-warm: self.residency.free()" in content
