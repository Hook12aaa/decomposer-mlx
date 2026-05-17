from PIL import Image, ImageDraw
import numpy as np

from decomposer.research.oracle import composite_layers, composite_ssim


def _solid_rgba(size, color):
    img = Image.new("RGBA", size, color)
    return img


def test_composite_layers_stacks_in_order():
    bg = _solid_rgba((64, 64), (200, 0, 0, 255))
    overlay = _solid_rgba((64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([16, 16, 48, 48], fill=(0, 200, 0, 255))
    composite = composite_layers([bg, overlay])
    px = composite.convert("RGB").getpixel((32, 32))
    assert px == (0, 200, 0)
    px = composite.convert("RGB").getpixel((0, 0))
    assert px == (200, 0, 0)


def test_composite_ssim_identical_images_is_1():
    a = Image.new("RGB", (64, 64), (100, 150, 200))
    score = composite_ssim(a, a)
    assert score > 0.99


def test_composite_ssim_resizes_to_match():
    a = Image.new("RGB", (64, 64), (100, 150, 200))
    b = Image.new("RGB", (128, 128), (100, 150, 200))
    score = composite_ssim(a, b)
    assert score > 0.99


from decomposer.research.oracle import per_layer_ssim_matched


def _layer(size, color):
    return Image.new("RGBA", size, color)


def test_per_layer_ssim_identical_layers_in_same_order():
    layers = [_layer((32, 32), (255, 0, 0, 255)),
              _layer((32, 32), (0, 255, 0, 255))]
    score, matching = per_layer_ssim_matched(layers, layers)
    assert score > 0.99
    assert matching == [0, 1]


def test_per_layer_ssim_handles_reorder():
    a = [_layer((32, 32), (255, 0, 0, 255)),
         _layer((32, 32), (0, 255, 0, 255))]
    b = [a[1], a[0]]
    score, matching = per_layer_ssim_matched(a, b)
    assert score > 0.99
    assert matching == [1, 0]


def test_per_layer_ssim_requires_same_length():
    a = [_layer((32, 32), (255, 0, 0, 255))]
    b = [_layer((32, 32), (255, 0, 0, 255)),
         _layer((32, 32), (0, 255, 0, 255))]
    import pytest
    with pytest.raises(ValueError, match="same number of layers"):
        per_layer_ssim_matched(a, b)


from decomposer.research.oracle import (
    QualityReport,
    check_non_degeneracy,
    score,
)


def test_check_non_degeneracy_flags_fully_transparent():
    transparent = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ok, reasons = check_non_degeneracy([transparent])
    assert ok is False
    assert any("opaque" in r for r in reasons)


def test_check_non_degeneracy_flags_fully_opaque():
    opaque = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
    ok, reasons = check_non_degeneracy([opaque])
    assert ok is False
    assert any("transparent" in r for r in reasons)


def test_check_non_degeneracy_accepts_mixed_alpha():
    img = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 16, 32], fill=(100, 100, 100, 255))
    ok, reasons = check_non_degeneracy([img])
    assert ok is True
    assert reasons == []


def test_score_returns_quality_report():
    bg = _layer((32, 32), (0, 0, 0, 0))
    draw = ImageDraw.Draw(bg)
    draw.rectangle([0, 0, 16, 32], fill=(200, 0, 0, 255))
    input_img = composite_layers([bg]).convert("RGB")
    report = score([bg], [bg], input_img)
    assert isinstance(report, QualityReport)
    assert report.composite_ssim > 0.99
    assert report.per_layer_ssim_matched > 0.99
    assert report.non_degenerate is True
    assert report.passes(composite_ssim_min=0.92, per_layer_ssim_min=0.85)
