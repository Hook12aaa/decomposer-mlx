import pytest
import torch
from PIL import Image

from decomposer.core.mps_backend import MpsBackend, _make_latent_output_pipeline_class
from decomposer.core.xray import Tracer


def _raise_unbound_local(name: str) -> None:
    src = f"def _f():\n    if False:\n        {name} = 1\n    return {name}\n"
    ns: dict = {}
    exec(src, ns)
    ns["_f"]()


def test_latent_output_pipeline_captures_latents_despite_upstream_bug():
    base_cls = _make_latent_output_pipeline_class()
    sentinel_latents = torch.randn(1, 64)

    class _Stub(base_cls):
        def __init__(self):
            pass

        def _invoke_parent_call(self_super, **kwargs):
            cb = kwargs["callback_on_step_end"]
            cb(self_super, 0, 0, {"latents": sentinel_latents})
            _raise_unbound_local("images")

    pipe = _Stub()
    out = pipe.denoise_only(image=None, prompt="x")
    assert out is sentinel_latents


def test_latent_output_pipeline_raises_if_callback_never_fires():
    base_cls = _make_latent_output_pipeline_class()

    class _Stub(base_cls):
        def __init__(self):
            pass

        def _invoke_parent_call(self_super, **kwargs):
            _raise_unbound_local("images")

    pipe = _Stub()
    with pytest.raises(RuntimeError, match="no latents were captured"):
        pipe.denoise_only(image=None, prompt="x")


def test_latent_output_pipeline_propagates_non_images_unbound_local():
    base_cls = _make_latent_output_pipeline_class()

    class _Stub(base_cls):
        def __init__(self):
            pass

        def _invoke_parent_call(self_super, **kwargs):
            _raise_unbound_local("something_else")

    pipe = _Stub()
    with pytest.raises(UnboundLocalError):
        pipe.denoise_only(image=None, prompt="x")


@pytest.mark.mps_required
def test_mps_backend_text_encoder_phase_runs():
    backend = MpsBackend()
    t = Tracer(run_id="r-te")
    cond = backend._encode_prompt(
        image=Image.new("RGB", (32, 32), (0, 0, 0)),
        prompt="a marketing advertisement",
        tracer=t,
    )
    assert cond is not None
    names = [s.name for s in t.report().stages]
    assert "load_text_encoder" in names
    assert "encode_prompt" in names
    assert "free_text_encoder" in names


@pytest.mark.mps_required
def test_mps_backend_decompose_returns_layers():
    backend = MpsBackend()
    img = Image.new("RGB", (64, 64), (200, 50, 50))
    t = Tracer(run_id="r-full")
    layers = backend.decompose(img, layers=3, resolution=640, steps=4, tracer=t)
    assert len(layers) == 3
    assert all(layer.mode == "RGBA" for layer in layers)
    rep = t.report()
    expected = [
        "load_text_encoder", "encode_prompt", "free_text_encoder",
        "load_vae", "encode_image_to_latent", "free_vae",
        "load_dit", "denoise_loop", "free_dit",
        "load_vae", "decode_layers", "free_vae",
    ]
    seen = [s.name for s in rep.stages]
    for name in expected:
        assert name in seen, f"missing stage: {name} (seen: {seen})"
