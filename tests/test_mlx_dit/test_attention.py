import mlx.core as mx
import pytest

from decomposer.mlx_dit.attention import MLXJointAttention
from decomposer.mlx_dit.rope import MLXRoPE3D
from decomposer.mlx_dit.transformer_block import MLXMMDiTBlock


def _make_rotary_emb(height: int, width: int, num_frames: int, axes_dims: list[int]):
    rope = MLXRoPE3D(axes_dims=axes_dims)
    img_cos, img_sin = rope.compute_freqs(height=height, width=width, num_frames=num_frames)
    img_seq_len = num_frames * height * width
    txt_cos = img_cos[:1].astype(mx.float32)
    txt_sin = img_sin[:1].astype(mx.float32)
    return (img_cos, img_sin), (txt_cos, txt_sin)


class TestMLXJointAttention:
    def test_produces_correct_shapes(self):
        dim, num_heads, head_dim = 64, 4, 16
        attn = MLXJointAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)

        img = mx.random.normal((1, 20, dim))
        txt = mx.random.normal((1, 10, dim))

        axes_dims = [4, 8, 8]
        (img_cos, img_sin), (txt_cos, txt_sin) = _make_rotary_emb(
            height=4, width=5, num_frames=1, axes_dims=axes_dims
        )

        txt_cos_expanded = mx.broadcast_to(txt_cos, (10, txt_cos.shape[-1]))
        txt_sin_expanded = mx.broadcast_to(txt_sin, (10, txt_sin.shape[-1]))

        image_rotary_emb = ((img_cos, img_sin), (txt_cos_expanded, txt_sin_expanded))
        img_out, txt_out = attn(img, txt, image_rotary_emb)

        assert img_out.shape == (1, 20, dim), f"Expected (1, 20, {dim}), got {img_out.shape}"
        assert txt_out.shape == (1, 10, dim), f"Expected (1, 10, {dim}), got {txt_out.shape}"

    def test_without_rope(self):
        dim, num_heads, head_dim = 64, 4, 16
        attn = MLXJointAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)

        img = mx.random.normal((1, 20, dim))
        txt = mx.random.normal((1, 10, dim))

        img_out, txt_out = attn(img, txt, image_rotary_emb=None)

        assert img_out.shape == (1, 20, dim)
        assert txt_out.shape == (1, 10, dim)

    def test_batch_size_two(self):
        dim, num_heads, head_dim = 64, 4, 16
        attn = MLXJointAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)

        img = mx.random.normal((2, 20, dim))
        txt = mx.random.normal((2, 10, dim))

        img_out, txt_out = attn(img, txt, image_rotary_emb=None)

        assert img_out.shape == (2, 20, dim)
        assert txt_out.shape == (2, 10, dim)


class TestMLXMMDiTBlock:
    def test_produces_correct_shapes(self):
        dim, num_heads, head_dim = 64, 4, 16
        block = MLXMMDiTBlock(dim=dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=4.0)

        img = mx.random.normal((1, 20, dim))
        txt = mx.random.normal((1, 10, dim))
        mod = mx.random.normal((1, dim))

        axes_dims = [4, 8, 8]
        (img_cos, img_sin), (txt_cos, txt_sin) = _make_rotary_emb(
            height=4, width=5, num_frames=1, axes_dims=axes_dims
        )
        txt_cos_expanded = mx.broadcast_to(txt_cos, (10, txt_cos.shape[-1]))
        txt_sin_expanded = mx.broadcast_to(txt_sin, (10, txt_sin.shape[-1]))
        image_rotary_emb = ((img_cos, img_sin), (txt_cos_expanded, txt_sin_expanded))

        img_out, txt_out = block(img, txt, mod, image_rotary_emb)

        assert img_out.shape == (1, 20, dim), f"Expected (1, 20, {dim}), got {img_out.shape}"
        assert txt_out.shape == (1, 10, dim), f"Expected (1, 10, {dim}), got {txt_out.shape}"

    def test_residual_connection(self):
        dim, num_heads, head_dim = 64, 4, 16
        block = MLXMMDiTBlock(dim=dim, num_heads=num_heads, head_dim=head_dim, mlp_ratio=4.0)

        img = mx.random.normal((1, 20, dim))
        txt = mx.random.normal((1, 10, dim))
        mod = mx.zeros((1, dim))

        img_out, txt_out = block(img, txt, mod, image_rotary_emb=None)

        assert img_out.shape == img.shape
        assert txt_out.shape == txt.shape
