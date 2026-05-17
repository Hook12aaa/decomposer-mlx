import numpy as np
import torch

from decomposer.core.gguf_loader import GgufLinear


def _pack_q8_0(weight: np.ndarray) -> bytes:
    flat = weight.reshape(-1)
    assert flat.size % 32 == 0
    out = bytearray()
    for i in range(0, flat.size, 32):
        block = flat[i:i + 32]
        scale = float(np.abs(block).max() / 127.0) if np.any(block) else 1e-8
        quant = np.clip(np.round(block / scale), -127, 127).astype(np.int8)
        out += np.float16(scale).tobytes()
        out += quant.tobytes()
    return bytes(out)


def test_gguf_linear_does_not_hold_full_precision_weight():
    """The core memory invariant: int8 quants + fp16 scales, no fp16 weight buffer."""
    rng = np.random.default_rng(42)
    weight = rng.standard_normal((128, 64)).astype(np.float32) * 0.1
    packed = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=packed, shape=(128, 64), bias=None,
                      dtype=torch.float16)

    buffer_names = {n for n, _ in layer.named_buffers()}
    assert "weight" not in buffer_names, (
        f"GgufLinear should not store a 'weight' buffer; that would defeat Q8. "
        f"Buffers found: {sorted(buffer_names)}"
    )
    assert "quants" in buffer_names
    assert "scales" in buffer_names


def test_gguf_linear_persistent_memory_is_below_fp16_equivalent():
    """Q8 persistent memory must be strictly less than the fp16 equivalent."""
    rng = np.random.default_rng(43)
    weight = rng.standard_normal((128, 64)).astype(np.float32) * 0.1
    packed = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=packed, shape=(128, 64), bias=None,
                      dtype=torch.float16)

    total_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in layer.buffers())

    fp16_equiv = 128 * 64 * 2
    assert total_bytes < fp16_equiv, (
        f"Q8 footprint {total_bytes} should be less than fp16 equivalent {fp16_equiv}"
    )


def test_gguf_linear_forward_produces_correct_shape():
    rng = np.random.default_rng(44)
    weight = rng.standard_normal((64, 32)).astype(np.float32) * 0.1
    packed = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=packed, shape=(64, 32), bias=None,
                      dtype=torch.float32)

    x = torch.randn(4, 32)
    out = layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()
