from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from decomposer.mlx_dit.attention import MLXJointAttention
from decomposer.mlx_dit.layers import MLXFeedForward, MLXLayerNorm


class MLXMMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int = 3072,
        num_heads: int = 24,
        head_dim: int = 128,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        mlp_hidden = int(dim * mlp_ratio)

        self.img_mod_silu = nn.SiLU()
        self.img_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.img_norm1 = MLXLayerNorm(dim, affine=False, eps=1e-6)
        self.attn = MLXJointAttention(dim=dim, num_heads=num_heads, head_dim=head_dim)
        self.img_norm2 = MLXLayerNorm(dim, affine=False, eps=1e-6)
        self.img_ff = MLXFeedForward(dim=dim, hidden_dim=mlp_hidden)

        self.txt_mod_silu = nn.SiLU()
        self.txt_mod_linear = nn.Linear(dim, 6 * dim, bias=True)
        self.txt_norm1 = MLXLayerNorm(dim, affine=False, eps=1e-6)
        self.txt_norm2 = MLXLayerNorm(dim, affine=False, eps=1e-6)
        self.txt_ff = MLXFeedForward(dim=dim, hidden_dim=mlp_hidden)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        text_embeddings: mx.array,
        image_rotary_emb: tuple,
    ) -> tuple[mx.array, mx.array]:
        img_mod_params = self.img_mod_linear(self.img_mod_silu(text_embeddings))
        txt_mod_params = self.txt_mod_linear(self.txt_mod_silu(text_embeddings))

        img_mod1, img_mod2 = mx.split(img_mod_params, 2, axis=-1)
        txt_mod1, txt_mod2 = mx.split(txt_mod_params, 2, axis=-1)

        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = _modulate(img_normed, img_mod1)

        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = _modulate(txt_normed, txt_mod1)

        img_attn_output, txt_attn_output = self.attn(
            img_modulated=img_modulated,
            txt_modulated=txt_modulated,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = _modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_ff(img_modulated2)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = _modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_ff(txt_modulated2)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        return hidden_states, encoder_hidden_states


def _modulate(x: mx.array, mod_params: mx.array) -> tuple[mx.array, mx.array]:
    shift, scale, gate = mx.split(mod_params, 3, axis=-1)
    return x * (1 + scale[:, None, :]) + shift[:, None, :], gate[:, None, :]
