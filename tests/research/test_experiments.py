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
