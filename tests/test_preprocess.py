import pytest
from PIL import Image

from decomposer.preprocess import PreprocessError, preprocess


def test_rgb_image_converted_to_rgba():
    img = Image.new("RGB", (256, 256), (100, 150, 200))
    result = preprocess(img)
    assert result.mode == "RGBA"
    assert result.size == (256, 256)


def test_rgba_image_passes_through():
    img = Image.new("RGBA", (512, 512), (100, 150, 200, 255))
    result = preprocess(img)
    assert result.mode == "RGBA"
    assert result.size == (512, 512)


def test_grayscale_converted_to_rgba():
    img = Image.new("L", (128, 128), 128)
    result = preprocess(img)
    assert result.mode == "RGBA"


def test_image_too_small_raises():
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    with pytest.raises(PreprocessError, match="64"):
        preprocess(img)


def test_image_exactly_64x64_passes():
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    result = preprocess(img)
    assert result.mode == "RGBA"


def test_oversized_image_logs_warning(caplog):
    img = Image.new("RGB", (4096, 2048), (0, 0, 0))
    import logging
    with caplog.at_level(logging.WARNING, logger="decomposer.preprocess"):
        result = preprocess(img)
    assert result.mode == "RGBA"
    assert any("2048" in r.message or "4096" in r.message for r in caplog.records)
