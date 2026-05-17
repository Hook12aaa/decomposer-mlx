import mlx.core as mx
import pytest

from decomposer.mlx_dit.rope import MLXRoPE3D


def test_rope_produces_correct_shapes():
    rope = MLXRoPE3D(axes_dims=[16, 56, 56])
    h, w = 46, 36
    cos_freqs, sin_freqs = rope.compute_freqs(height=h, width=w, num_frames=4)
    total_half_dim = (16 + 56 + 56) // 2
    assert cos_freqs.shape == (4 * h * w, total_half_dim)
    assert sin_freqs.shape == (4 * h * w, total_half_dim)
    assert cos_freqs.ndim == 2


def test_rope_apply_preserves_shape():
    rope = MLXRoPE3D(axes_dims=[16, 56, 56])
    q = mx.random.normal((1, 24, 100, 128))
    freqs = rope.compute_freqs(height=10, width=10, num_frames=1)
    out = rope.apply(q, freqs)
    assert out.shape == q.shape
