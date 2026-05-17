import mlx.core as mx
import mlx.nn as nn
import pytest

from decomposer.mlx_dit.layers import (
    MLXRMSNorm,
    MLXLayerNorm,
    MLXFeedForward,
    MLXTimestepEmbedding,
    MLXTimesteps,
)


def test_rms_norm_preserves_shape():
    norm = MLXRMSNorm(64)
    x = mx.random.normal((2, 10, 64))
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_layer_norm_preserves_shape():
    norm = MLXLayerNorm(64, affine=False)
    x = mx.random.normal((2, 10, 64))
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_feed_forward_projects():
    ff = MLXFeedForward(dim=64, hidden_dim=256)
    x = mx.random.normal((2, 10, 64))
    out = ff(x)
    assert out.shape == (2, 10, 64)


def test_timestep_embedding_produces_embed():
    te = MLXTimestepEmbedding(in_dim=256, hidden_dim=512, out_dim=64)
    t = mx.random.normal((2, 256))
    out = te(t)
    assert out.shape == (2, 64)


def test_timesteps_produces_sinusoidal():
    ts = MLXTimesteps(num_channels=256)
    t = mx.array([0.1, 0.5])
    out = ts(t)
    assert out.shape == (2, 256)
