import json

from PIL import Image

from decomposer.workflow import run_workflow


def test_workflow_produces_layers_and_manifest(tmp_path):
    img = Image.new("RGB", (256, 256), (100, 150, 200))
    src = tmp_path / "input.jpg"
    img.save(src)
    out = tmp_path / "output"
    result = run_workflow(
        image_path=src, output_dir=out, layers=3, resolution=640,
        steps=8, seed=42, backend_name="fake", trace=False,
    )
    assert result.success
    assert len(result.layer_files) == 3
    assert all(f.exists() for f in result.layer_files)
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["source"] == "input.jpg"
    assert manifest["backend"] == "fake"
    assert manifest["seed"] == 42
    assert len(manifest["layers"]) == 3
    assert manifest["layers"][0]["classification"] in [
        "background", "hero_image", "text", "logo",
        "cta_button", "overlay", "unknown",
    ]

def test_workflow_rejects_tiny_image(tmp_path):
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    src = tmp_path / "tiny.png"
    img.save(src)
    result = run_workflow(
        image_path=src, output_dir=tmp_path / "out", layers=3,
        resolution=640, steps=8, backend_name="fake",
    )
    assert not result.success
    assert "64" in result.error

def test_workflow_creates_output_dir(tmp_path):
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    src = tmp_path / "img.png"
    img.save(src)
    out = tmp_path / "nested" / "output"
    result = run_workflow(
        image_path=src, output_dir=out, layers=3, resolution=640,
        steps=8, backend_name="fake",
    )
    assert result.success
    assert out.exists()

def test_workflow_includes_quality_warnings(tmp_path):
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    src = tmp_path / "img.png"
    img.save(src)
    result = run_workflow(
        image_path=src, output_dir=tmp_path / "out", layers=3,
        resolution=640, steps=8, backend_name="fake",
    )
    assert result.success
    manifest = json.loads(result.manifest_path.read_text())
    assert "quality_warnings" in manifest

def test_workflow_with_trace(tmp_path):
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    src = tmp_path / "img.png"
    img.save(src)
    out = tmp_path / "out"
    result = run_workflow(
        image_path=src, output_dir=out, layers=3, resolution=640,
        steps=8, backend_name="fake", trace=True,
    )
    assert result.success
    assert (out / "trace.json").exists()

def test_workflow_nonexistent_image(tmp_path):
    result = run_workflow(
        image_path=tmp_path / "nope.jpg", output_dir=tmp_path / "out",
        layers=3, resolution=640, steps=8, backend_name="fake",
    )
    assert not result.success
    assert "not found" in result.error.lower() or "No such file" in result.error


def test_workflow_preflight_mlx_missing_weights(tmp_path, monkeypatch):
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    src = tmp_path / "img.png"
    img.save(src)
    monkeypatch.setattr(
        "decomposer.workflow.get_settings",
        lambda: type("S", (), {"mlx_weights_dir": tmp_path / "no-weights"})(),
    )
    result = run_workflow(
        image_path=src, output_dir=tmp_path / "out", layers=3,
        resolution=640, steps=8, backend_name="mlx",
    )
    assert not result.success
    assert "convert-to-mlx" in result.error
