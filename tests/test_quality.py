import numpy as np
from PIL import Image

from decomposer.quality import validate_layers


_rng = np.random.default_rng(42)


def _make_layer(w: int, h: int, alpha: int = 255) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = _rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    arr[..., 3] = alpha
    return Image.fromarray(arr, "RGBA")

def test_all_opaque_layers_warns_non_degeneracy():
    layers = [_make_layer(100, 100, alpha=255) for _ in range(3)]
    warnings = validate_layers(layers)
    assert any(w.code == "all_opaque" for w in warnings)

def test_diverse_layers_no_warning():
    bg = _make_layer(100, 100, alpha=255)
    fg = _make_layer(100, 100, alpha=0)
    arr = np.asarray(fg)
    arr2 = arr.copy()
    arr2[20:80, 20:80, 3] = 255
    arr2[20:80, 20:80, :3] = [255, 0, 0]
    fg2 = Image.fromarray(arr2, "RGBA")
    warnings = validate_layers([bg, fg2])
    assert not any(w.code == "all_opaque" for w in warnings)

def test_all_transparent_layer_warns():
    layers = [_make_layer(100, 100, alpha=0)]
    warnings = validate_layers(layers)
    assert any(w.code == "empty_layer" for w in warnings)

def test_identical_layers_warns_low_diversity():
    layer = _make_layer(100, 100, alpha=255)
    layers = [layer.copy(), layer.copy(), layer.copy()]
    warnings = validate_layers(layers)
    assert any(w.code == "low_diversity" for w in warnings)

def test_warning_has_message():
    layers = [_make_layer(100, 100, alpha=0)]
    warnings = validate_layers(layers)
    assert len(warnings) > 0
    assert len(warnings[0].message) > 0
