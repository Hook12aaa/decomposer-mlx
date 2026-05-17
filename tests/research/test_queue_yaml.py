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
