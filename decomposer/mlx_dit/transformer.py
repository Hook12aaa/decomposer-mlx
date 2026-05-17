from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from decomposer.mlx_dit.layers import (
    MLXLayerNorm,
    MLXRMSNorm,
    MLXTimestepEmbedding,
    MLXTimesteps,
)
from decomposer.mlx_dit.rope import MLXRoPE3D
from decomposer.mlx_dit.transformer_block import MLXMMDiTBlock


class MLXAdaLayerNormContinuous(nn.Module):
    def __init__(self, embedding_dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.silu = nn.SiLU()
        self.linear = nn.Linear(conditioning_dim, embedding_dim * 2, bias=True)
        self.norm = MLXLayerNorm(embedding_dim, affine=False, eps=1e-6)

    def __call__(self, x: mx.array, conditioning: mx.array) -> mx.array:
        emb = self.linear(self.silu(conditioning))
        scale = emb[:, : self.embedding_dim]
        shift = emb[:, self.embedding_dim :]
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class MLXTimeTextEmbed(nn.Module):
    def __init__(
        self,
        timestep_proj_dim: int = 256,
        inner_dim: int = 3072,
        use_additional_t_cond: bool = False,
    ) -> None:
        super().__init__()
        self.time_proj = MLXTimesteps(num_channels=timestep_proj_dim)
        self.timestep_embedder = MLXTimestepEmbedding(
            in_dim=timestep_proj_dim, hidden_dim=inner_dim, out_dim=inner_dim
        )
        self.use_additional_t_cond = use_additional_t_cond
        if use_additional_t_cond:
            self.addition_t_embedding = nn.Embedding(2, inner_dim)

    def __call__(
        self,
        timestep: mx.array,
        hidden_states: mx.array,
        addition_t_cond: mx.array | None = None,
    ) -> mx.array:
        timesteps_proj = self.time_proj(timestep)
        conditioning = self.timestep_embedder(
            timesteps_proj.astype(hidden_states.dtype)
        )
        if self.use_additional_t_cond:
            if addition_t_cond is None:
                addition_t_cond = mx.zeros((timestep.shape[0],), dtype=mx.int32)
            addition_t_emb = self.addition_t_embedding(addition_t_cond)
            conditioning = conditioning + addition_t_emb.astype(conditioning.dtype)
        return conditioning


class MLXQwenTransformer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_layers = config["num_layers"]
        num_heads = config["num_attention_heads"]
        head_dim = config["attention_head_dim"]
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        joint_attention_dim = config["joint_attention_dim"]
        axes_dims_rope = config["axes_dims_rope"]
        patch_size = config["patch_size"]

        self.inner_dim = num_heads * head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels

        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_norm = MLXRMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.time_text_embed = MLXTimeTextEmbed(
            timestep_proj_dim=256,
            inner_dim=self.inner_dim,
            use_additional_t_cond=config.get("use_additional_t_cond", False),
        )

        self.pos_embed = MLXRoPE3D(axes_dims=axes_dims_rope)

        self.transformer_blocks = [
            MLXMMDiTBlock(
                dim=self.inner_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                mlp_ratio=4.0,
            )
            for _ in range(num_layers)
        ]

        self.norm_out = MLXAdaLayerNormContinuous(self.inner_dim, self.inner_dim)
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * out_channels
        )

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        height: int,
        width: int,
        num_frames: int = 1,
    ) -> mx.array:
        hidden_states = self.img_in(hidden_states)

        batch_size = hidden_states.shape[0]
        timestep_broadcast = mx.broadcast_to(timestep, (batch_size,)).astype(
            hidden_states.dtype
        )

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        text_embeddings = self.time_text_embed(timestep_broadcast, hidden_states)

        image_rotary_emb = self._compute_rotary_emb(
            hidden_states, encoder_hidden_states, height, width, num_frames
        )

        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                text_embeddings=text_embeddings,
                image_rotary_emb=image_rotary_emb,
            )

        hidden_states = self.norm_out(hidden_states, text_embeddings)
        hidden_states = self.proj_out(hidden_states)

        return hidden_states

    def _compute_rotary_emb(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        height: int,
        width: int,
        num_frames: int,
    ) -> tuple:
        img_cos, img_sin, max_vid_index = self.pos_embed.compute_freqs(
            height=height, width=width, num_frames=num_frames
        )

        txt_seq_len = encoder_hidden_states.shape[1]
        all_freqs = self.pos_embed.pos_freqs
        txt_cos_np = all_freqs[max_vid_index : max_vid_index + txt_seq_len, :, 0]
        txt_sin_np = all_freqs[max_vid_index : max_vid_index + txt_seq_len, :, 1]
        txt_cos = mx.array(txt_cos_np.astype(np.float32))
        txt_sin = mx.array(txt_sin_np.astype(np.float32))

        return (img_cos, img_sin), (txt_cos, txt_sin)
