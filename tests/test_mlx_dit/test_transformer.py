import mlx.core as mx
import pytest

from decomposer.mlx_dit.transformer import MLXQwenTransformer


class TestMLXQwenTransformer:
    def test_forward_produces_output(self):
        config = {
            "num_layers": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 16,
            "in_channels": 64,
            "out_channels": 16,
            "joint_attention_dim": 128,
            "axes_dims_rope": [4, 8, 8],
            "patch_size": 2,
        }
        model = MLXQwenTransformer(config)
        latent = mx.random.normal((1, 20, 64))
        cond = mx.random.normal((1, 10, 128))
        timestep = mx.array([0.5])

        out = model(latent, cond, timestep, height=4, width=5, num_frames=1)

        assert out.shape[0] == 1
        assert out.shape[1] == 20
        assert out.shape[-1] == 64

    def test_multiple_layers(self):
        config = {
            "num_layers": 4,
            "num_attention_heads": 4,
            "attention_head_dim": 16,
            "in_channels": 32,
            "out_channels": 8,
            "joint_attention_dim": 64,
            "axes_dims_rope": [4, 8, 8],
            "patch_size": 2,
        }
        model = MLXQwenTransformer(config)
        latent = mx.random.normal((1, 12, 32))
        cond = mx.random.normal((1, 8, 64))
        timestep = mx.array([0.1])

        out = model(latent, cond, timestep, height=3, width=4, num_frames=1)

        assert out.shape == (1, 12, 32)

    def test_batch_size_two(self):
        config = {
            "num_layers": 2,
            "num_attention_heads": 4,
            "attention_head_dim": 16,
            "in_channels": 64,
            "out_channels": 16,
            "joint_attention_dim": 128,
            "axes_dims_rope": [4, 8, 8],
            "patch_size": 2,
        }
        model = MLXQwenTransformer(config)
        latent = mx.random.normal((2, 20, 64))
        cond = mx.random.normal((2, 10, 128))
        timestep = mx.array([0.5])

        out = model(latent, cond, timestep, height=4, width=5, num_frames=1)

        assert out.shape == (2, 20, 64)
