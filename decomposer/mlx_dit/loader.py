"""Load converted MLX safetensors into the MLX Qwen transformer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from decomposer.mlx_dit.transformer import MLXQwenTransformer

logger = logging.getLogger(__name__)


def load_mlx_transformer(
    weights_dir: Path,
    bits: int = 4,
    group_size: int = 64,
) -> MLXQwenTransformer:
    """Load a quantized MLX transformer from a weights directory.

    The directory must contain ``config.json`` (diffusers transformer config)
    and ``weights.safetensors`` (output of the GGUF-to-MLX converter).

    Parameters
    ----------
    weights_dir : directory containing config.json and weights.safetensors
    bits : quantization bit width used during conversion
    group_size : quantization group size used during conversion

    Returns
    -------
    MLXQwenTransformer with quantized weights loaded and evaluated.
    """
    weights_dir = Path(weights_dir)
    config = json.loads((weights_dir / "config.json").read_text())
    bits = config.get("mlx_bits", bits)
    group_size = config.get("mlx_group_size", group_size)
    model = MLXQwenTransformer(config)

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda _, m: isinstance(m, nn.Linear) and not isinstance(m, nn.Embedding),
    )

    shard_files = sorted(weights_dir.glob("weights-*.safetensors"))
    if not shard_files:
        shard_files = [weights_dir / "weights.safetensors"]
    all_weights: list[tuple[str, mx.array]] = []
    for sf in shard_files:
        shard = mx.load(str(sf))
        all_weights.extend(shard.items())
        logger.info("Loaded shard %s (%d entries)", sf.name, len(shard))
    model.load_weights(all_weights)

    mx.eval(model.parameters())
    logger.info(
        "MLX transformer loaded from %s: %d weight entries, %d-bit quantized",
        weights_dir,
        len(all_weights),
        bits,
    )
    return model
