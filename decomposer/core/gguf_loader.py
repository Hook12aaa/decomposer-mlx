import numpy as np
import torch
import torch.nn as nn


Q8_BLOCK_SIZE = 32
Q8_BLOCK_BYTES = 2 + Q8_BLOCK_SIZE

QK_K = 256
K_SCALE_SIZE = 12
Q4_K_TYPE_SIZE = 144
Q5_K_TYPE_SIZE = 176


def _unpack_q8_0_to_tensors(packed: bytes, shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (quants:int8[n_blocks, 32], scales:fp16[n_blocks]), kept separate so persistent memory matches Q8 footprint."""
    n_elements = int(np.prod(shape))
    assert n_elements % Q8_BLOCK_SIZE == 0
    n_blocks = n_elements // Q8_BLOCK_SIZE
    assert len(packed) == n_blocks * Q8_BLOCK_BYTES

    arr = np.frombuffer(packed, dtype=np.uint8).reshape(n_blocks, Q8_BLOCK_BYTES)
    scales_np = np.ascontiguousarray(arr[:, :2]).view(np.float16).reshape(n_blocks)
    quants_np = np.ascontiguousarray(arr[:, 2:]).view(np.int8)
    return torch.from_numpy(quants_np.copy()), torch.from_numpy(scales_np.copy())


def _get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode packed scales/mins from K_SCALE_SIZE bytes per super-block.

    Input: uint8 tensor of shape (n_blocks, K_SCALE_SIZE=12).
    Returns: (sc, m) each of shape (n_blocks, 8) as uint8.
    """
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape(n_blocks, 3, 4)
    d, m, m_d = torch.split(scales, 1, dim=-2)
    d = d.squeeze(-2)
    m = m.squeeze(-2)
    m_d = m_d.squeeze(-2)

    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape(n_blocks, 8), mn.reshape(n_blocks, 8)


def _dequantize_q4_k_blocks(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q4_K super-blocks to float.

    Input: uint8 tensor of shape (n_blocks, Q4_K_TYPE_SIZE=144).
    Returns: float tensor of shape (n_blocks, QK_K=256).
    """
    n_blocks = blocks.shape[0]
    d_bytes = blocks[:, :2].contiguous()
    dmin_bytes = blocks[:, 2:4].contiguous()
    scales_bytes = blocks[:, 4:4 + K_SCALE_SIZE].contiguous()
    qs = blocks[:, 4 + K_SCALE_SIZE:].contiguous()

    d = d_bytes.view(torch.float16).to(dtype)
    dmin = dmin_bytes.view(torch.float16).to(dtype)

    sc, m = _get_scale_min(scales_bytes)
    sc = sc.to(dtype)
    m = m.to(dtype)

    d_scaled = (d * sc).reshape(n_blocks, -1, 1)
    dm = (dmin * m).reshape(n_blocks, -1, 1)

    qs = qs.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1, 32)

    return (d_scaled * qs.to(dtype) - dm).reshape(n_blocks, QK_K)


def _dequantize_q5_k_blocks(blocks: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize Q5_K super-blocks to float.

    Input: uint8 tensor of shape (n_blocks, Q5_K_TYPE_SIZE=176).
    Returns: float tensor of shape (n_blocks, QK_K=256).
    """
    n_blocks = blocks.shape[0]
    d_bytes = blocks[:, :2].contiguous()
    dmin_bytes = blocks[:, 2:4].contiguous()
    scales_bytes = blocks[:, 4:4 + K_SCALE_SIZE].contiguous()
    qh = blocks[:, 4 + K_SCALE_SIZE:4 + K_SCALE_SIZE + QK_K // 8].contiguous()
    qs = blocks[:, 4 + K_SCALE_SIZE + QK_K // 8:].contiguous()

    d = d_bytes.view(torch.float16).to(dtype)
    dmin = dmin_bytes.view(torch.float16).to(dtype)

    sc, m = _get_scale_min(scales_bytes)
    sc = sc.to(dtype)
    m = m.to(dtype)

    d_scaled = (d * sc).reshape(n_blocks, -1, 1)
    dm = (dmin * m).reshape(n_blocks, -1, 1)

    ql = qs.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        [0, 4], device=blocks.device, dtype=torch.uint8
    ).reshape(1, 1, 2, 1)
    qh_expanded = qh.reshape(n_blocks, -1, 1, 32) >> torch.tensor(
        list(range(8)), device=blocks.device, dtype=torch.uint8
    ).reshape(1, 1, 8, 1)
    ql = (ql & 0x0F).reshape(n_blocks, -1, 32)
    qh_expanded = (qh_expanded & 0x01).reshape(n_blocks, -1, 32)
    q = ql | (qh_expanded << 4)

    return (d_scaled * q.to(dtype) - dm).reshape(n_blocks, QK_K)


def _unpack_kquant_to_raw(packed: bytes, shape: tuple[int, ...], type_size: int) -> torch.Tensor:
    """Store raw packed bytes as uint8 tensor for k-quant formats.

    Returns uint8 tensor of shape (n_blocks, type_size). Persistent memory
    matches on-disk footprint.
    """
    n_elements = int(np.prod(shape))
    assert n_elements % QK_K == 0, f"element count {n_elements} not divisible by QK_K={QK_K}"
    n_blocks = n_elements // QK_K
    assert len(packed) == n_blocks * type_size, (
        f"packed size {len(packed)} != expected {n_blocks * type_size} "
        f"(n_blocks={n_blocks}, type_size={type_size})"
    )
    arr = np.frombuffer(packed, dtype=np.uint8).reshape(n_blocks, type_size)
    return torch.from_numpy(arr.copy())


class GgufLinear(nn.Module):
    """Linear layer storing Q8_0 weights persistently; dequantizes per-forward.

    Persistent memory is int8 quants + fp16 scales (~ on-disk Q8 footprint). The
    full-precision weight matrix is materialized only inside forward() and
    discarded immediately, so model-wide memory stays at Q8 scale even across
    thousands of layers. Materializing it as a buffer would defeat the point
    of Q8 storage on a 48 GB unified-memory system.
    """

    def __init__(
        self,
        *,
        quantized_weight: bytes,
        shape: tuple[int, int],
        bias: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self._shape = shape
        self._dtype = dtype
        quants, scales = _unpack_q8_0_to_tensors(quantized_weight, shape)
        self.register_buffer("quants", quants)
        self.register_buffer("scales", scales)
        self.bias = nn.Parameter(bias.to(dtype)) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = (
            self.quants.to(self._dtype) * self.scales.to(self._dtype).unsqueeze(-1)
        ).view(*self._shape)
        out = x.to(self._dtype) @ weight.T
        if self.bias is not None:
            out = out + self.bias
        return out


class GgufKQuantLinear(nn.Module):
    """Linear layer storing Q4_K or Q5_K weights persistently; dequantizes per-forward.

    Persistent memory is the raw packed bytes (uint8 tensor matching on-disk
    footprint). The full-precision weight matrix is materialized only inside
    forward() and discarded immediately.

    Q4_K_M persistent footprint: ~4.5 bits/param (144 bytes / 256 elements).
    Q5_K_M persistent footprint: ~5.5 bits/param (176 bytes / 256 elements).
    Both are significantly smaller than Q8_0 (~8.5 bits/param).
    """

    def __init__(
        self,
        *,
        quantized_weight: bytes,
        shape: tuple[int, int],
        bias: torch.Tensor | None,
        dtype: torch.dtype,
        quant_type: str,
    ) -> None:
        super().__init__()
        self._shape = shape
        self._dtype = dtype
        self._quant_type = quant_type

        if quant_type == "Q4_K":
            type_size = Q4_K_TYPE_SIZE
        elif quant_type == "Q5_K":
            type_size = Q5_K_TYPE_SIZE
        else:
            raise ValueError(f"Unsupported k-quant type: {quant_type!r}")

        self._type_size = type_size
        raw = _unpack_kquant_to_raw(quantized_weight, shape, type_size)
        self.register_buffer("packed", raw)
        self.bias = nn.Parameter(bias.to(dtype)) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._quant_type == "Q4_K":
            weight = _dequantize_q4_k_blocks(self.packed, self._dtype)
        else:
            weight = _dequantize_q5_k_blocks(self.packed, self._dtype)
        weight = weight.reshape(*self._shape)
        out = x.to(self._dtype) @ weight.T
        if self.bias is not None:
            out = out + self.bias
        return out
