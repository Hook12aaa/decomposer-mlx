import mlx.core as mx
import numpy as np
import pytest

from decomposer.mlx_dit.scheduler import (
    compute_mu,
    flow_match_euler_step,
    get_sigmas,
    get_timesteps,
)


class TestScheduler:
    def test_get_sigmas_length(self):
        sigmas = get_sigmas(8, mu=1.0)
        assert len(sigmas) == 9
        assert sigmas[0] > 0.9
        assert sigmas[-1] == 0.0

    def test_get_sigmas_monotonically_decreasing(self):
        sigmas = get_sigmas(8, mu=1.0)
        for i in range(len(sigmas) - 1):
            assert sigmas[i] > sigmas[i + 1]

    def test_get_timesteps_length(self):
        sigmas = get_sigmas(8, mu=1.0)
        ts = get_timesteps(sigmas)
        assert len(ts) == 8

    def test_compute_mu_positive(self):
        mu = compute_mu(1024)
        assert mu > 0

    def test_euler_step_shape(self):
        latent = mx.random.normal((1, 100, 64))
        pred = mx.random.normal((1, 100, 64))
        out = flow_match_euler_step(latent, pred, 1.0, 0.5)
        assert out.shape == latent.shape

    def test_euler_step_identity_at_zero_dt(self):
        latent = mx.random.normal((1, 50, 32))
        pred = mx.random.normal((1, 50, 32))
        out = flow_match_euler_step(latent, pred, 0.5, 0.5)
        np.testing.assert_allclose(
            np.array(out), np.array(latent), atol=1e-6
        )


class TestMlxBackendImport:
    def test_can_import(self):
        from decomposer.core.mlx_backend import MlxBackend
        assert MlxBackend is not None

    def test_pack_latents_shape(self):
        from decomposer.core.mlx_backend import _pack_latents_np
        latents = np.random.randn(1, 4, 16, 20, 18).astype(np.float32)
        packed = _pack_latents_np(latents, 1, 16, 20, 18, 4)
        assert packed.shape == (1, 4 * 10 * 9, 64)


@pytest.mark.mps_required
class TestMlxBackendIntegration:
    def test_decompose_produces_layers(self):
        from PIL import Image
        from decomposer.core.mlx_backend import MlxBackend
        from decomposer.core.xray import Tracer

        backend = MlxBackend()
        img = Image.new("RGB", (64, 64), (200, 50, 50))
        t = Tracer(run_id="r-mlx")
        result = backend.decompose(img, layers=3, resolution=640, steps=4, tracer=t)
        assert len(result) == 3
        assert all(layer.mode == "RGBA" for layer in result)
        rep = t.report()
        seen = [s.name for s in rep.stages]
        for expected in ["load_dit", "denoise_loop"]:
            assert expected in seen, f"missing stage: {expected}"
