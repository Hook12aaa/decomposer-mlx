"""GGUF to MLX weight converter.

Reads a GGUF file containing the Qwen-Image-Layered transformer weights,
dequantizes each tensor to fp32, then re-quantizes weight matrices to MLX 8-bit
format (by default) and stores non-weight tensors (norms, biases, embeddings) as fp16.

Output: a directory containing ``weights.safetensors`` and ``config.json``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GGUF name → MLX model parameter name mapping
# ---------------------------------------------------------------------------
# The GGUF file uses diffusers state-dict names (after stripping
# ``model.diffusion_model.`` prefix).  The MLX model in
# ``decomposer.mlx_dit.transformer`` uses its own naming scheme.
#
# This mapping was built by dumping both name lists and aligning them.
# ---------------------------------------------------------------------------

def build_name_mapping() -> dict[str, str]:
    """Return dict mapping GGUF tensor names → MLX parameter names."""
    mapping: dict[str, str] = {}

    # Top-level projections
    mapping["img_in.weight"] = "img_in.weight"
    mapping["img_in.bias"] = "img_in.bias"
    mapping["txt_in.weight"] = "txt_in.weight"
    mapping["txt_in.bias"] = "txt_in.bias"
    mapping["txt_norm.weight"] = "txt_norm.weight"

    # Output norm (AdaLayerNormContinuous → norm_out.linear)
    mapping["norm_out.linear.weight"] = "norm_out.linear.weight"
    mapping["norm_out.linear.bias"] = "norm_out.linear.bias"

    # Output projection
    mapping["proj_out.weight"] = "proj_out.weight"
    mapping["proj_out.bias"] = "proj_out.bias"

    # Time-text embedding: timestep projection MLP
    mapping["time_text_embed.timestep_embedder.linear_1.weight"] = (
        "time_text_embed.timestep_embedder.linear1.weight"
    )
    mapping["time_text_embed.timestep_embedder.linear_1.bias"] = (
        "time_text_embed.timestep_embedder.linear1.bias"
    )
    mapping["time_text_embed.timestep_embedder.linear_2.weight"] = (
        "time_text_embed.timestep_embedder.linear2.weight"
    )
    mapping["time_text_embed.timestep_embedder.linear_2.bias"] = (
        "time_text_embed.timestep_embedder.linear2.bias"
    )

    # Addition-t embedding (nn.Embedding for additional_t_cond)
    mapping["time_text_embed.addition_t_embedding.weight"] = (
        "time_text_embed.addition_t_embedding.weight"
    )

    # Per-block mappings (60 blocks)
    for i in range(60):
        gp = f"transformer_blocks.{i}"
        mp = f"transformer_blocks.{i}"

        # Attention projections: image stream
        mapping[f"{gp}.attn.to_q.weight"] = f"{mp}.attn.to_q.weight"
        mapping[f"{gp}.attn.to_q.bias"] = f"{mp}.attn.to_q.bias"
        mapping[f"{gp}.attn.to_k.weight"] = f"{mp}.attn.to_k.weight"
        mapping[f"{gp}.attn.to_k.bias"] = f"{mp}.attn.to_k.bias"
        mapping[f"{gp}.attn.to_v.weight"] = f"{mp}.attn.to_v.weight"
        mapping[f"{gp}.attn.to_v.bias"] = f"{mp}.attn.to_v.bias"
        mapping[f"{gp}.attn.to_out.0.weight"] = f"{mp}.attn.attn_to_out.0.weight"
        mapping[f"{gp}.attn.to_out.0.bias"] = f"{mp}.attn.attn_to_out.0.bias"

        # Attention projections: text stream
        mapping[f"{gp}.attn.add_q_proj.weight"] = f"{mp}.attn.add_q_proj.weight"
        mapping[f"{gp}.attn.add_q_proj.bias"] = f"{mp}.attn.add_q_proj.bias"
        mapping[f"{gp}.attn.add_k_proj.weight"] = f"{mp}.attn.add_k_proj.weight"
        mapping[f"{gp}.attn.add_k_proj.bias"] = f"{mp}.attn.add_k_proj.bias"
        mapping[f"{gp}.attn.add_v_proj.weight"] = f"{mp}.attn.add_v_proj.weight"
        mapping[f"{gp}.attn.add_v_proj.bias"] = f"{mp}.attn.add_v_proj.bias"
        mapping[f"{gp}.attn.to_add_out.weight"] = f"{mp}.attn.to_add_out.weight"
        mapping[f"{gp}.attn.to_add_out.bias"] = f"{mp}.attn.to_add_out.bias"

        # Attention QK norms
        mapping[f"{gp}.attn.norm_q.weight"] = f"{mp}.attn.norm_q.weight"
        mapping[f"{gp}.attn.norm_k.weight"] = f"{mp}.attn.norm_k.weight"
        mapping[f"{gp}.attn.norm_added_q.weight"] = f"{mp}.attn.norm_added_q.weight"
        mapping[f"{gp}.attn.norm_added_k.weight"] = f"{mp}.attn.norm_added_k.weight"

        # Image modulation: diffusers Sequential(SiLU, Linear) → our img_mod_linear
        mapping[f"{gp}.img_mod.1.weight"] = f"{mp}.img_mod_linear.weight"
        mapping[f"{gp}.img_mod.1.bias"] = f"{mp}.img_mod_linear.bias"

        # Text modulation
        mapping[f"{gp}.txt_mod.1.weight"] = f"{mp}.txt_mod_linear.weight"
        mapping[f"{gp}.txt_mod.1.bias"] = f"{mp}.txt_mod_linear.bias"

        # Image FFN: diffusers net.0.proj / net.2 → our img_ff.linear1 / img_ff.linear2
        mapping[f"{gp}.img_mlp.net.0.proj.weight"] = f"{mp}.img_ff.linear1.weight"
        mapping[f"{gp}.img_mlp.net.0.proj.bias"] = f"{mp}.img_ff.linear1.bias"
        mapping[f"{gp}.img_mlp.net.2.weight"] = f"{mp}.img_ff.linear2.weight"
        mapping[f"{gp}.img_mlp.net.2.bias"] = f"{mp}.img_ff.linear2.bias"

        # Text FFN
        mapping[f"{gp}.txt_mlp.net.0.proj.weight"] = f"{mp}.txt_ff.linear1.weight"
        mapping[f"{gp}.txt_mlp.net.0.proj.bias"] = f"{mp}.txt_ff.linear1.bias"
        mapping[f"{gp}.txt_mlp.net.2.weight"] = f"{mp}.txt_ff.linear2.weight"
        mapping[f"{gp}.txt_mlp.net.2.bias"] = f"{mp}.txt_ff.linear2.bias"

    return mapping


# ---------------------------------------------------------------------------
# Tensor reading helpers
# ---------------------------------------------------------------------------

def _read_tensor_as_fp32(tensor) -> np.ndarray:
    """Read a GGUF tensor and return it as a fp32 numpy array."""
    shape = tuple(int(x) for x in reversed(tensor.shape.tolist()))
    raw = tensor.data.tobytes()
    t = tensor.tensor_type

    if t == GGMLQuantizationType.F32:
        return np.frombuffer(raw, dtype=np.float32).reshape(shape).copy()

    if t == GGMLQuantizationType.F16:
        return np.frombuffer(raw, dtype=np.float16).reshape(shape).astype(np.float32).copy()

    if t == GGMLQuantizationType.BF16:
        import torch
        u16 = np.frombuffer(raw, dtype=np.uint16).reshape(shape).copy()
        return torch.from_numpy(u16).view(torch.bfloat16).float().numpy()

    if t == GGMLQuantizationType.Q8_0:
        from decomposer.core.gguf_loader import _unpack_q8_0_to_tensors
        import torch
        n_elements = int(np.prod(shape))
        block_size = 32
        n_blocks = n_elements // block_size
        byte_size = n_blocks * (2 + block_size)
        packed = np.asarray(tensor.data, dtype=np.uint8).tobytes()[:byte_size]
        q, s = _unpack_q8_0_to_tensors(packed, shape)
        return (q.float() * s.float().unsqueeze(-1)).reshape(*shape).numpy()

    raise RuntimeError(f"Unsupported tensor type {t} for {tensor.name}")


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

GGUF_TENSOR_PREFIX = "model.diffusion_model."


def convert_gguf_to_mlx(
    gguf_path: str | Path,
    output_dir: Path,
    bits: int = 8,
    group_size: int = 64,
) -> None:
    """Convert a Qwen-Image-Layered GGUF file to MLX quantized safetensors.

    Parameters
    ----------
    gguf_path : path to the GGUF file
    output_dir : directory for output weights.safetensors + config.json
    bits : quantization bit width (default 4)
    group_size : quantization group size (default 64)
    """
    from diffusers import QwenImageTransformer2DModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write config
    config = dict(
        QwenImageTransformer2DModel.load_config(
            "Qwen/Qwen-Image-Layered", subfolder="transformer"
        )
    )
    config["mlx_bits"] = bits
    config["mlx_group_size"] = group_size
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str)
    )
    logger.info("Config written to %s", output_dir / "config.json")

    # Index GGUF
    reader = GGUFReader(str(gguf_path))
    gguf_index: dict[str, object] = {}
    has_prefix = any(
        t.name.startswith(GGUF_TENSOR_PREFIX) for t in reader.tensors
    )
    for t in reader.tensors:
        name = t.name
        if has_prefix and name.startswith(GGUF_TENSOR_PREFIX):
            name = name[len(GGUF_TENSOR_PREFIX):]
        gguf_index[name] = t
    logger.info("GGUF indexed: %d tensors", len(gguf_index))

    # Build name mapping
    name_map = build_name_mapping()

    # Verify coverage
    unmapped_gguf = set(gguf_index.keys()) - set(name_map.keys())
    if unmapped_gguf:
        logger.warning(
            "Unmapped GGUF tensors (will be skipped): %s",
            sorted(unmapped_gguf),
        )

    missing_gguf = set(name_map.keys()) - set(gguf_index.keys())
    if missing_gguf:
        logger.warning(
            "Expected GGUF tensors not found: %s",
            sorted(missing_gguf),
        )

    # Convert in shards to stay within memory budget
    SHARD_SIZE = 400
    shard_weights: dict[str, mx.array] = {}
    converted = 0
    shard_idx = 0
    total = sum(1 for n in name_map if n in gguf_index)

    for gguf_name, mlx_name in name_map.items():
        if gguf_name not in gguf_index:
            continue
        tensor = gguf_index[gguf_name]
        arr = _read_tensor_as_fp32(tensor)
        mlx_arr = mx.array(arr)
        del arr

        is_large_weight = (
            "weight" in mlx_name
            and mlx_arr.ndim == 2
            and mlx_arr.shape[0] >= 64
            and mlx_arr.shape[1] >= 64
        )
        if is_large_weight:
            q_weight, q_scales, q_biases = mx.quantize(
                mlx_arr, group_size=group_size, bits=bits
            )
            mx.eval(q_weight, q_scales, q_biases)
            shard_weights[mlx_name] = q_weight
            shard_weights[mlx_name.replace(".weight", ".scales")] = q_scales
            shard_weights[mlx_name.replace(".weight", ".biases")] = q_biases
        else:
            fp16 = mlx_arr.astype(mx.float16)
            mx.eval(fp16)
            shard_weights[mlx_name] = fp16
        del mlx_arr

        converted += 1
        if converted % 50 == 0:
            logger.info("Converted %d/%d tensors...", converted, total)

        if len(shard_weights) >= SHARD_SIZE:
            shard_file = output_dir / f"weights-{shard_idx:04d}.safetensors"
            mx.save_safetensors(str(shard_file), shard_weights)
            logger.info("Saved shard %d (%d entries) to %s", shard_idx, len(shard_weights), shard_file.name)
            shard_weights.clear()
            import gc; gc.collect()
            shard_idx += 1

    if shard_weights:
        shard_file = output_dir / f"weights-{shard_idx:04d}.safetensors"
        mx.save_safetensors(str(shard_file), shard_weights)
        logger.info("Saved shard %d (%d entries) to %s", shard_idx, len(shard_weights), shard_file.name)
        shard_weights.clear()
        shard_idx += 1

    logger.info(
        "Saved %d tensors across %d shards to %s",
        converted, shard_idx, output_dir,
    )
