import numpy as np
from PIL import Image

from decomposer.classifier import classify_layers


def _make_layer(w: int, h: int, alpha_fill: int = 255, color: tuple = (100, 100, 100)) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = color
    arr[..., 3] = alpha_fill
    return Image.fromarray(arr, "RGBA")


def _make_partial_layer(
    w: int, h: int, x0: int, y0: int, x1: int, y1: int,
    color: tuple = (200, 50, 50), alpha: int = 255,
) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[y0:y1, x0:x1, :3] = color
    arr[y0:y1, x0:x1, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def test_full_opaque_layer_classified_as_background():
    layer = _make_layer(640, 480, alpha_fill=255)
    infos = classify_layers([layer])
    assert len(infos) == 1
    assert infos[0].classification == "background"


def test_empty_layer_has_zero_coverage():
    layer = _make_layer(640, 480, alpha_fill=0)
    infos = classify_layers([layer])
    assert infos[0].alpha_coverage == 0.0


def test_partial_layer_has_correct_bounding_box():
    layer = _make_partial_layer(640, 480, 100, 50, 300, 200)
    infos = classify_layers([layer])
    assert infos[0].bounding_box == (100, 50, 300, 200)


def test_small_compact_element_classified_as_logo():
    bg = _make_layer(640, 480, alpha_fill=255)
    logo = _make_partial_layer(640, 480, 10, 10, 70, 50, color=(255, 0, 0))
    infos = classify_layers([bg, logo])
    assert infos[1].classification == "logo"


def test_mid_coverage_large_area_classified_as_hero():
    bg = _make_layer(640, 480, alpha_fill=255)
    hero = _make_partial_layer(640, 480, 50, 50, 500, 400, color=(100, 200, 150))
    infos = classify_layers([bg, hero])
    assert infos[1].classification == "hero_image"


def test_classify_returns_correct_mean_rgb():
    layer = _make_layer(100, 100, alpha_fill=255, color=(50, 100, 150))
    infos = classify_layers([layer])
    assert infos[0].mean_rgb == (50, 100, 150)


def test_classify_returns_file_index():
    layers = [_make_layer(100, 100) for _ in range(3)]
    infos = classify_layers(layers)
    assert [i.index for i in infos] == [0, 1, 2]


def test_confidence_is_between_zero_and_one():
    layer = _make_layer(640, 480, alpha_fill=255)
    infos = classify_layers([layer])
    assert 0.0 <= infos[0].confidence <= 1.0
