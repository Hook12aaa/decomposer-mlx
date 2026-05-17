import json

from decomposer.classifier import LayerInfo
from decomposer.manifest import LayerEntry, Manifest, write_manifest


def test_manifest_serializes_to_json():
    m = Manifest(
        source="banner.jpg",
        source_dimensions=(1920, 1080),
        resolution=640,
        steps=8,
        layers_requested=3,
        backend="mlx",
        seed=None,
        wall_time_seconds=300.5,
        quality_warnings=[],
        layers=[
            LayerEntry(
                file="layer_0.png",
                index=0,
                classification="background",
                confidence=0.92,
                alpha_coverage=1.0,
                bounding_box=(0, 0, 640, 480),
                mean_rgb=(45, 82, 130),
                file_size_bytes=1148576,
            ),
        ],
    )
    data = json.loads(m.model_dump_json())
    assert data["source"] == "banner.jpg"
    assert data["layers"][0]["classification"] == "background"
    assert data["seed"] is None


def test_write_manifest_creates_file(tmp_path):
    m = Manifest(
        source="test.png",
        source_dimensions=(100, 100),
        resolution=640,
        steps=8,
        layers_requested=1,
        backend="mlx",
        seed=42,
        wall_time_seconds=10.0,
        quality_warnings=[],
        layers=[],
    )
    write_manifest(m, tmp_path / "manifest.json")
    loaded = json.loads((tmp_path / "manifest.json").read_text())
    assert loaded["source"] == "test.png"
    assert loaded["seed"] == 42


def test_layer_entry_from_layer_info():
    info = LayerInfo(
        index=0,
        classification="background",
        confidence=0.95,
        alpha_coverage=1.0,
        bounding_box=(0, 0, 640, 480),
        mean_rgb=(50, 100, 150),
    )
    entry = LayerEntry.from_layer_info(info, file="layer_0.png", file_size_bytes=12345)
    assert entry.file == "layer_0.png"
    assert entry.file_size_bytes == 12345
    assert entry.classification == "background"
