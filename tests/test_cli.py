from unittest.mock import patch

from typer.testing import CliRunner

from decomposer.cli import app

runner = CliRunner()


def test_doctor_fake_backend_passes():
    result = runner.invoke(app, ["doctor", "--fake"])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output or "OK" in result.output


def test_doctor_fake_skips_hf_auth_check():
    with patch("decomposer.cli._check_hf_auth") as mock_check:
        result = runner.invoke(app, ["doctor", "--fake"])
        assert result.exit_code == 0
        mock_check.assert_not_called()


def test_doctor_real_path_fails_clearly_when_hf_repo_unreachable():
    with patch("decomposer.cli._check_hf_auth", return_value=False):
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 1


def test_decompose_with_fake_writes_layers(tmp_path):
    from PIL import Image
    src = tmp_path / "in.png"
    Image.new("RGB", (64, 64), (0, 0, 0)).save(src)
    out = tmp_path / "out"
    result = runner.invoke(app, ["decompose", str(src), "--layers", "3",
                                  "--out", str(out), "--fake"])
    assert result.exit_code == 0, result.output
    pngs = list(out.glob("layer_*.png"))
    assert len(pngs) == 3


def test_run_with_fake_produces_manifest(tmp_path):
    from PIL import Image
    src = tmp_path / "photo.jpg"
    Image.new("RGB", (256, 256), (100, 150, 200)).save(src)
    out = tmp_path / "out"
    result = runner.invoke(app, [
        "run", str(src), "--layers", "3", "-o", str(out), "--backend", "fake",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "manifest.json").exists()
    import json
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source"] == "photo.jpg"
    assert len(manifest["layers"]) == 3

def test_run_rejects_tiny_image(tmp_path):
    from PIL import Image
    src = tmp_path / "tiny.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(src)
    result = runner.invoke(app, [
        "run", str(src), "--backend", "fake",
    ])
    assert result.exit_code == 1

def test_run_batch_multiple_images(tmp_path):
    from PIL import Image
    src1 = tmp_path / "a.jpg"
    src2 = tmp_path / "b.jpg"
    Image.new("RGB", (128, 128), (100, 0, 0)).save(src1)
    Image.new("RGB", (128, 128), (0, 100, 0)).save(src2)
    out = tmp_path / "out"
    result = runner.invoke(app, [
        "run", str(src1), str(src2), "-o", str(out), "--backend", "fake",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "a" / "manifest.json").exists()
    assert (out / "b" / "manifest.json").exists()

def test_run_default_output_dir(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "img.png"
    Image.new("RGB", (128, 128), (0, 0, 0)).save(src)
    result = runner.invoke(app, [
        "run", str(src), "--backend", "fake",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "output" / "img" / "manifest.json").exists()
