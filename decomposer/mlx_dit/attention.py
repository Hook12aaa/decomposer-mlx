from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.core.fast import scaled_dot_product_attention


class MLXJointAttention(nn.Module):
    def __init__(self, dim: int = 3072, num_heads: int = 24, head_dim: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.add_q_proj = nn.Linear(dim, dim)
        self.add_k_proj = nn.Linear(dim, dim)
        self.add_v_proj = nn.Linear(dim, dim)

        self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_q = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.norm_added_k = nn.RMSNorm(self.head_dim, eps=1e-6)

        self.attn_to_out = [nn.Linear(dim, dim)]
        self.to_add_out = nn.Linear(dim, dim)

    def __call__(
        self,
        img_modulated: mx.array,
        txt_modulated: mx.array,
        image_rotary_emb: tuple,
    ) -> tuple[mx.array, mx.array]:
        img_query = self.to_q(img_modulated)
        img_key = self.to_k(img_modulated)
        img_value = self.to_v(img_modulated)

        txt_query = self.add_q_proj(txt_modulated)
        txt_key = self.add_k_proj(txt_modulated)
        txt_value = self.add_v_proj(txt_modulated)

        img_query = mx.reshape(img_query, (img_query.shape[0], img_query.shape[1], self.num_heads, self.head_dim))
        img_key = mx.reshape(img_key, (img_key.shape[0], img_key.shape[1], self.num_heads, self.head_dim))
        img_value = mx.reshape(img_value, (img_value.shape[0], img_value.shape[1], self.num_heads, self.head_dim))

        txt_query = mx.reshape(txt_query, (txt_query.shape[0], txt_query.shape[1], self.num_heads, self.head_dim))
        txt_key = mx.reshape(txt_key, (txt_key.shape[0], txt_key.shape[1], self.num_heads, self.head_dim))
        txt_value = mx.reshape(txt_value, (txt_value.shape[0], txt_value.shape[1], self.num_heads, self.head_dim))

        img_query = self.norm_q(img_query)
        img_key = self.norm_k(img_key)
        txt_query = self.norm_added_q(txt_query)
        txt_key = self.norm_added_k(txt_key)

        if image_rotary_emb is not None:
            (img_cos, img_sin), (txt_cos, txt_sin) = image_rotary_emb
            img_query = _apply_rope(img_query, img_cos, img_sin)
            img_key = _apply_rope(img_key, img_cos, img_sin)
            txt_query = _apply_rope(txt_query, txt_cos, txt_sin)
            txt_key = _apply_rope(txt_key, txt_cos, txt_sin)

        joint_query = mx.concatenate([txt_query, img_query], axis=1)
        joint_key = mx.concatenate([txt_key, img_key], axis=1)
        joint_value = mx.concatenate([txt_value, img_value], axis=1)

        query_bhsd = mx.transpose(joint_query, (0, 2, 1, 3))
        key_bhsd = mx.transpose(joint_key, (0, 2, 1, 3))
        value_bhsd = mx.transpose(joint_value, (0, 2, 1, 3))

        scale_value = 1.0 / (self.head_dim ** 0.5)
        hidden_states_bhsd = scaled_dot_product_attention(
            query_bhsd, key_bhsd, value_bhsd, scale=scale_value
        )

        hidden_states = mx.transpose(hidden_states_bhsd, (0, 2, 1, 3))
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        hidden_states = mx.reshape(hidden_states, (batch_size, seq_len, self.num_heads * self.head_dim))
        hidden_states = hidden_states.astype(joint_query.dtype)

        seq_txt = txt_modulated.shape[1]
        txt_attn_output = hidden_states[:, :seq_txt, :]
        img_attn_output = hidden_states[:, seq_txt:, :]

        img_attn_output = self.attn_to_out[0](img_attn_output)
        txt_attn_output = self.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


def _apply_rope(x: mx.array, cos_vals: mx.array, sin_vals: mx.array) -> mx.array:
    x_float = x.astype(mx.float32)
    x_reshaped = mx.reshape(x_float, (*x.shape[:-1], -1, 2))

    x_real = x_reshaped[..., 0]
    x_imag = x_reshaped[..., 1]

    freqs_cos = cos_vals[None, :, None, :]
    freqs_sin = sin_vals[None, :, None, :]

    if freqs_cos.shape[-1] != x_real.shape[-1]:
        freqs_cos = freqs_cos[..., : x_real.shape[-1]]
        freqs_sin = freqs_sin[..., : x_real.shape[-1]]

    out_real = x_real * freqs_cos - x_imag * freqs_sin
    out_imag = x_real * freqs_sin + x_imag * freqs_cos

    out_pairs = mx.stack([out_real, out_imag], axis=-1)
    x_out = mx.reshape(out_pairs, (*x.shape[:-1], -1))

    return x_out.astype(x.dtype)
