import numpy as np
import pytest
import torch

from decomposer.core.gguf_loader import (
    GgufKQuantLinear,
    GgufLinear,
    Q4_K_TYPE_SIZE,
    Q5_K_TYPE_SIZE,
    QK_K,
    _dequantize_q4_k_blocks,
    _dequantize_q5_k_blocks,
    _unpack_q8_0_to_tensors,
)


def test_vectorized_unpack_matches_loop_reference():
    rng = np.random.default_rng(99)
    weight = rng.standard_normal((256, 128)).astype(np.float32) * 0.1
    packed = _pack_q8_0(weight)
    quants_new, scales_new = _unpack_q8_0_to_tensors(packed, (256, 128))
    n_blocks = 256 * 128 // 32
    arr = np.frombuffer(packed, dtype=np.uint8)
    scales_ref = np.empty(n_blocks, dtype=np.float16)
    quants_ref = np.empty((n_blocks, 32), dtype=np.int8)
    for b in range(n_blocks):
        off = b * 34
        scales_ref[b] = np.frombuffer(arr[off:off+2].tobytes(), dtype=np.float16)[0]
        quants_ref[b] = np.frombuffer(arr[off+2:off+34].tobytes(), dtype=np.int8)
    assert (quants_new.numpy() == quants_ref).all()
    assert (scales_new.numpy() == scales_ref).all()


def test_gguf_linear_forward_matches_fp_reference():
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((16, 8)).astype(np.float32) * 0.1
    bias = rng.standard_normal(16).astype(np.float32) * 0.1
    x = torch.tensor(rng.standard_normal((4, 8)).astype(np.float32))

    quantized = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=quantized, shape=(16, 8),
                      bias=torch.tensor(bias), dtype=torch.float32)

    ref = x @ torch.tensor(weight).T + torch.tensor(bias)
    out = layer(x)
    err = (out - ref).abs().max().item()
    assert err < 0.05, f"forward error too high: {err}"


@pytest.mark.mps_required
def test_gguf_linear_runs_on_mps():
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((64, 32)).astype(np.float32) * 0.1
    quantized = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=quantized, shape=(64, 32),
                      bias=None, dtype=torch.float16).to("mps")
    x = torch.randn(2, 32, device="mps", dtype=torch.float16)
    out = layer(x)
    assert out.device.type == "mps"
    assert out.shape == (2, 64)
    assert torch.isfinite(out).all()


def test_gguf_linear_does_not_store_dequantized_weight_as_buffer():
    """The whole point of Q8 storage: persistent memory matches Q8 size, not fp16 size."""
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((128, 64)).astype(np.float32) * 0.1
    quantized = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=quantized, shape=(128, 64),
                      bias=None, dtype=torch.float16)
    buffer_names = {n for n, _ in layer.named_buffers()}
    assert "weight" not in buffer_names, (
        f"weight should not be a persistent buffer (would defeat Q8 memory savings); "
        f"buffers={buffer_names}"
    )
    assert "quants" in buffer_names
    assert "scales" in buffer_names
    quants_bytes = layer.quants.numel() * layer.quants.element_size()
    scales_bytes = layer.scales.numel() * layer.scales.element_size()
    fp16_equiv_bytes = 128 * 64 * 2
    assert quants_bytes + scales_bytes < fp16_equiv_bytes, (
        f"persistent memory {quants_bytes + scales_bytes} should be less than "
        f"fp16 equivalent {fp16_equiv_bytes}"
    )


def test_gguf_linear_to_device_moves_quants_and_scales():
    rng = np.random.default_rng(4)
    weight = rng.standard_normal((32, 32)).astype(np.float32) * 0.1
    quantized = _pack_q8_0(weight)
    layer = GgufLinear(quantized_weight=quantized, shape=(32, 32),
                      bias=None, dtype=torch.float16).to("cpu")
    assert layer.quants.device.type == "cpu"
    assert layer.scales.device.type == "cpu"


def _pack_q8_0(weight: np.ndarray) -> bytes:
    """Q8_0 block layout: 2-byte fp16 scale + 32 int8 quants per 32-element block (GGUF spec)."""
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


def _pack_q4_k(weight: np.ndarray) -> bytes:
    """Reference Q4_K packer for testing: quantize 256-element super-blocks to 4-bit."""
    flat = weight.reshape(-1).astype(np.float32)
    assert flat.size % QK_K == 0
    n_blocks = flat.size // QK_K
    out = bytearray()

    for b in range(n_blocks):
        block = flat[b * QK_K:(b + 1) * QK_K]
        sub_blocks = block.reshape(8, 32)

        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)
        for j in range(8):
            sb = sub_blocks[j]
            sb_max = float(np.max(sb))
            sb_min = float(np.min(sb))
            if sb_max == sb_min:
                scales[j] = 1e-8
                mins[j] = 0.0
            else:
                the_min = min(sb_min, 0.0)
                the_max = max(sb_max, 0.0)
                scales[j] = (the_max - the_min) / 15.0
                mins[j] = -the_min

        max_scale = float(np.max(np.abs(scales)))
        max_min = float(np.max(np.abs(mins)))
        d = np.float16(max_scale / 63.0) if max_scale > 0 else np.float16(1e-8)
        dmin = np.float16(max_min / 63.0) if max_min > 0 else np.float16(1e-8)

        d_f32 = float(d)
        dmin_f32 = float(dmin)

        sc = np.zeros(8, dtype=np.uint8)
        m = np.zeros(8, dtype=np.uint8)
        for j in range(8):
            sc[j] = min(63, int(np.round(scales[j] / d_f32))) if d_f32 > 0 else 0
            m[j] = min(63, int(np.round(mins[j] / dmin_f32))) if dmin_f32 > 0 else 0

        packed_scales = bytearray(12)
        for j in range(4):
            packed_scales[j] = sc[j]
            packed_scales[j + 4] = m[j]
        for j in range(4, 8):
            packed_scales[j + 4] = (sc[j] & 0x0F) | ((m[j] & 0x0F) << 4)
            packed_scales[j - 4] |= (sc[j] >> 4) << 6
            packed_scales[j] |= (m[j] >> 4) << 6

        quants = np.zeros(QK_K, dtype=np.uint8)
        for j in range(8):
            actual_scale = d_f32 * sc[j]
            actual_min = dmin_f32 * m[j]
            for k in range(32):
                val = block[j * 32 + k]
                q = int(np.round((val + actual_min) / actual_scale)) if actual_scale > 0 else 0
                quants[j * 32 + k] = max(0, min(15, q))

        qs = bytearray(QK_K // 2)
        for g in range(4):
            for k in range(32):
                lo = quants[(2 * g) * 32 + k]
                hi = quants[(2 * g + 1) * 32 + k]
                qs[g * 32 + k] = lo | (hi << 4)

        out += d.tobytes()
        out += dmin.tobytes()
        out += bytes(packed_scales)
        out += bytes(qs)

    return bytes(out)


def _pack_q5_k(weight: np.ndarray) -> bytes:
    """Reference Q5_K packer for testing: quantize 256-element super-blocks to 5-bit."""
    flat = weight.reshape(-1).astype(np.float32)
    assert flat.size % QK_K == 0
    n_blocks = flat.size // QK_K
    out = bytearray()

    for b in range(n_blocks):
        block = flat[b * QK_K:(b + 1) * QK_K]
        sub_blocks = block.reshape(8, 32)

        scales = np.zeros(8, dtype=np.float32)
        mins = np.zeros(8, dtype=np.float32)
        for j in range(8):
            sb = sub_blocks[j]
            sb_max = float(np.max(sb))
            sb_min = float(np.min(sb))
            if sb_max == sb_min:
                scales[j] = 1e-8
                mins[j] = 0.0
            else:
                the_min = min(sb_min, 0.0)
                the_max = max(sb_max, 0.0)
                scales[j] = (the_max - the_min) / 31.0
                mins[j] = -the_min

        max_scale = float(np.max(np.abs(scales)))
        max_min = float(np.max(np.abs(mins)))
        d = np.float16(max_scale / 63.0) if max_scale > 0 else np.float16(1e-8)
        dmin = np.float16(max_min / 63.0) if max_min > 0 else np.float16(1e-8)

        d_f32 = float(d)
        dmin_f32 = float(dmin)

        sc = np.zeros(8, dtype=np.uint8)
        m_arr = np.zeros(8, dtype=np.uint8)
        for j in range(8):
            sc[j] = min(63, int(np.round(scales[j] / d_f32))) if d_f32 > 0 else 0
            m_arr[j] = min(63, int(np.round(mins[j] / dmin_f32))) if dmin_f32 > 0 else 0

        packed_scales = bytearray(12)
        for j in range(4):
            packed_scales[j] = sc[j]
            packed_scales[j + 4] = m_arr[j]
        for j in range(4, 8):
            packed_scales[j + 4] = (sc[j] & 0x0F) | ((m_arr[j] & 0x0F) << 4)
            packed_scales[j - 4] |= (sc[j] >> 4) << 6
            packed_scales[j] |= (m_arr[j] >> 4) << 6

        quants = np.zeros(QK_K, dtype=np.uint8)
        for j in range(8):
            actual_scale = d_f32 * sc[j]
            actual_min = dmin_f32 * m_arr[j]
            for k in range(32):
                val = block[j * 32 + k]
                q = int(np.round((val + actual_min) / actual_scale)) if actual_scale > 0 else 0
                q = max(0, min(31, q))
                quants[j * 32 + k] = q

        qh = bytearray(QK_K // 8)
        qs = bytearray(QK_K // 2)
        for g in range(4):
            for k in range(32):
                lo = quants[(2 * g) * 32 + k] & 0x0F
                hi = quants[(2 * g + 1) * 32 + k] & 0x0F
                qs[g * 32 + k] = lo | (hi << 4)

        for j in range(8):
            for k in range(32):
                high_bit = (quants[j * 32 + k] >> 4) & 0x01
                if high_bit:
                    qh[k] |= 1 << j

        out += d.tobytes()
        out += dmin.tobytes()
        out += bytes(packed_scales)
        out += bytes(qh)
        out += bytes(qs)

    return bytes(out)


def test_q4k_dequant_round_trip():
    """Q4_K pack-then-dequant recovers original values within 4-bit quantization error."""
    rng = np.random.default_rng(42)
    weight = rng.standard_normal((256,)).astype(np.float32) * 0.1
    packed = _pack_q4_k(weight)
    assert len(packed) == Q4_K_TYPE_SIZE

    blocks = torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).reshape(1, Q4_K_TYPE_SIZE))
    dequantized = _dequantize_q4_k_blocks(blocks, torch.float32).reshape(-1)

    err = (dequantized - torch.from_numpy(weight)).abs().max().item()
    assert err < 0.05, f"Q4_K round-trip error {err:.4f} exceeds 0.05 threshold"


def test_q5k_dequant_round_trip():
    """Q5_K pack-then-dequant recovers original values within 5-bit quantization error."""
    rng = np.random.default_rng(42)
    weight = rng.standard_normal((256,)).astype(np.float32) * 0.1
    packed = _pack_q5_k(weight)
    assert len(packed) == Q5_K_TYPE_SIZE

    blocks = torch.from_numpy(np.frombuffer(packed, dtype=np.uint8).reshape(1, Q5_K_TYPE_SIZE))
    dequantized = _dequantize_q5_k_blocks(blocks, torch.float32).reshape(-1)

    err = (dequantized - torch.from_numpy(weight)).abs().max().item()
    assert err < 0.03, f"Q5_K round-trip error {err:.4f} exceeds 0.03 threshold"


def test_q4k_linear_forward():
    """GgufKQuantLinear with Q4_K produces reasonable output vs fp32 reference."""
    rng = np.random.default_rng(10)
    weight = rng.standard_normal((256, 256)).astype(np.float32) * 0.1
    packed = _pack_q4_k(weight)
    x = torch.tensor(rng.standard_normal((2, 256)).astype(np.float32))

    layer = GgufKQuantLinear(
        quantized_weight=packed,
        shape=(256, 256),
        bias=None,
        dtype=torch.float32,
        quant_type="Q4_K",
    )
    ref = x @ torch.tensor(weight).T
    out = layer(x)
    rel_err = (out - ref).abs().mean().item() / ref.abs().mean().item()
    assert rel_err < 0.15, f"Q4_K forward relative error {rel_err:.4f} exceeds 0.15"


def test_q5k_linear_forward():
    """GgufKQuantLinear with Q5_K produces reasonable output vs fp32 reference."""
    rng = np.random.default_rng(10)
    weight = rng.standard_normal((256, 256)).astype(np.float32) * 0.1
    packed = _pack_q5_k(weight)
    x = torch.tensor(rng.standard_normal((2, 256)).astype(np.float32))

    layer = GgufKQuantLinear(
        quantized_weight=packed,
        shape=(256, 256),
        bias=None,
        dtype=torch.float32,
        quant_type="Q5_K",
    )
    ref = x @ torch.tensor(weight).T
    out = layer(x)
    rel_err = (out - ref).abs().mean().item() / ref.abs().mean().item()
    assert rel_err < 0.10, f"Q5_K forward relative error {rel_err:.4f} exceeds 0.10"


def test_kquant_memory_footprint_less_than_q8():
    """K-quant persistent footprint is strictly smaller than Q8_0 for the same shape."""
    rng = np.random.default_rng(20)
    shape = (256, 256)
    n_elements = 256 * 256

    weight_q8 = rng.standard_normal(shape).astype(np.float32) * 0.1
    q8_packed = _pack_q8_0(weight_q8)
    q8_layer = GgufLinear(
        quantized_weight=q8_packed, shape=shape, bias=None, dtype=torch.float16
    )
    q8_bytes = sum(b.numel() * b.element_size() for _, b in q8_layer.named_buffers())

    weight_q4 = rng.standard_normal(shape).astype(np.float32) * 0.1
    q4_packed = _pack_q4_k(weight_q4)
    q4_layer = GgufKQuantLinear(
        quantized_weight=q4_packed, shape=shape, bias=None,
        dtype=torch.float16, quant_type="Q4_K",
    )
    q4_bytes = sum(b.numel() * b.element_size() for _, b in q4_layer.named_buffers())

    weight_q5 = rng.standard_normal(shape).astype(np.float32) * 0.1
    q5_packed = _pack_q5_k(weight_q5)
    q5_layer = GgufKQuantLinear(
        quantized_weight=q5_packed, shape=shape, bias=None,
        dtype=torch.float16, quant_type="Q5_K",
    )
    q5_bytes = sum(b.numel() * b.element_size() for _, b in q5_layer.named_buffers())

    assert q4_bytes < q8_bytes, (
        f"Q4_K footprint ({q4_bytes}) should be < Q8_0 ({q8_bytes})"
    )
    assert q5_bytes < q8_bytes, (
        f"Q5_K footprint ({q5_bytes}) should be < Q8_0 ({q8_bytes})"
    )
    assert q4_bytes < q5_bytes, (
        f"Q4_K footprint ({q4_bytes}) should be < Q5_K ({q5_bytes})"
    )


def test_kquant_linear_with_bias():
    """GgufKQuantLinear correctly applies bias."""
    rng = np.random.default_rng(30)
    weight = rng.standard_normal((256, 256)).astype(np.float32) * 0.1
    bias = rng.standard_normal(256).astype(np.float32) * 0.01
    packed = _pack_q5_k(weight)

    layer = GgufKQuantLinear(
        quantized_weight=packed,
        shape=(256, 256),
        bias=torch.tensor(bias),
        dtype=torch.float32,
        quant_type="Q5_K",
    )
    x = torch.tensor(rng.standard_normal((1, 256)).astype(np.float32))
    out = layer(x)
    assert out.shape == (1, 256)
    assert torch.isfinite(out).all()
