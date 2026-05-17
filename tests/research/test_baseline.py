from pathlib import Path
from PIL import Image
from decomposer.research.baseline import Baseline, save_baseline, load_baseline


def test_save_and_load_baseline_roundtrip(tmp_path):
    layers = [Image.new("RGBA", (32, 32), (i * 50, 0, 0, 255)) for i in range(3)]
    trace = {"run_id": "bl-1", "stages": [], "total_wall_ms": 100.0}
    image = Image.new("RGB", (32, 32), (100, 100, 100))
    baseline = Baseline(
        run_id="bl-1",
        input_image=image,
        layers=layers,
        trace=trace,
        layers_count=3,
        resolution=640,
        steps=8,
        commit_sha="abc123",
    )
    save_baseline(baseline, tmp_path)
    loaded = load_baseline(tmp_path)
    assert loaded.run_id == "bl-1"
    assert len(loaded.layers) == 3
    assert loaded.layers_count == 3
    assert loaded.trace["run_id"] == "bl-1"
    assert loaded.commit_sha == "abc123"
