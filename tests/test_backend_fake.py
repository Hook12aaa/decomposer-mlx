from PIL import Image
from decomposer.core.backend import FakeBackend
from decomposer.core.xray import Tracer


def test_fake_backend_returns_n_rgba_layers():
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    backend = FakeBackend(latency_ms=10)
    layers = backend.decompose(img, layers=6, resolution=256, steps=4)
    assert len(layers) == 6
    for layer in layers:
        assert layer.mode == "RGBA"
        assert layer.size == (256, 256)


def test_fake_backend_records_to_tracer():
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    backend = FakeBackend(latency_ms=5)
    t = Tracer(run_id="run-x")
    backend.decompose(img, layers=4, resolution=128, steps=4, tracer=t)
    rep = t.report()
    names = [s.name for s in rep.stages]
    for expected in ["load_text_encoder", "encode_prompt", "load_dit",
                     "denoise_loop", "load_vae", "decode_layers"]:
        assert expected in names
