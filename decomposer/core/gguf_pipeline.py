"""GGUF loader for the Qwen-Image-Layered transformer.

Loads `QwenImageTransformer2DModel` from its diffusers config on the `meta`
device (no weight allocation), then walks the module tree and:

- For every `nn.Linear`, swaps in a quantized layer backed by the matching
  GGUF block: `GgufLinear` for Q8_0, `GgufKQuantLinear` for Q4_K/Q5_K,
  or `Bf16Linear` for BF16/F16/F32 tensors.
- For every other tensor parameter (norms, embeddings, biases) reads the
  fp16/fp32/bf16 tensor directly out of the GGUF file.

Persistent footprint matches the on-disk quantization level: ~21 GB for Q8_0,
~15 GB for Q5_K_M, ~13 GB for Q4_K_M, rather than the fp16 inflation (~40 GB).
"""

from __future__ import annotations

import gc
import hashlib
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gguf import GGMLQuantizationType, GGUFReader, ReaderTensor

from decomposer.core.gguf_loader import GgufKQuantLinear, GgufLinear

logger = logging.getLogger(__name__)


GGUF_TENSOR_PREFIX = "model.diffusion_model."


class Bf16Linear(nn.Module):
    """Linear backed by a bf16 weight read directly from the GGUF.

    Used for the handful of input/output projection layers (img_in, txt_in,
    norm_out, proj_out, timestep MLP) that GGUF authors keep in BF16 rather
    than Q8_0. Subclassing nn.Module (not nn.Linear) so the "no stock
    nn.Linear remains" invariant of the Q8 loader still holds.
    """

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        compute_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
        self._compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(self._compute_dtype)
        out = x.to(self._compute_dtype) @ w.T
        if self.bias is not None:
            out = out + self.bias.to(self._compute_dtype)
        return out


def _logical_shape(tensor: ReaderTensor) -> tuple[int, ...]:
    """GGUF stores shape in reverse order (innermost dim first). Convert to torch order."""
    return tuple(int(x) for x in reversed(tensor.shape.tolist()))


def _read_dequantized(tensor: ReaderTensor) -> torch.Tensor:
    """Read a non-Q8 tensor (norms, embeddings, biases) as a torch tensor.

    Q8_0 tensors are handled separately because we never materialize their
    full-precision form during loading.
    """
    shape = _logical_shape(tensor)
    data = tensor.data
    t = tensor.tensor_type
    # GGUFReader stores F16/F32 as already-typed numpy arrays but leaves BF16
    # (and quants) as raw uint8 memmaps; normalize via .tobytes() then reinterpret.
    raw_bytes = data.tobytes()
    if t == GGMLQuantizationType.F32:
        arr = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)
        return torch.from_numpy(arr.copy())
    if t == GGMLQuantizationType.F16:
        arr = np.frombuffer(raw_bytes, dtype=np.float16).reshape(shape)
        return torch.from_numpy(arr.copy())
    if t == GGMLQuantizationType.BF16:
        u16 = np.frombuffer(raw_bytes, dtype=np.uint16).reshape(shape)
        return torch.from_numpy(u16.copy()).view(torch.bfloat16)
    raise RuntimeError(
        f"Tensor {tensor.name!r} has unsupported type {t}; expected F32/F16/BF16 for "
        f"non-Linear tensors. Refusing to silently dequantize."
    )


_QUANTIZED_TYPES = frozenset({
    GGMLQuantizationType.Q8_0,
    GGMLQuantizationType.Q4_K,
    GGMLQuantizationType.Q5_K,
})


def _raw_quant_bytes(tensor: ReaderTensor) -> bytes:
    """Extract the raw byte payload for any supported quantized tensor type."""
    if tensor.tensor_type not in _QUANTIZED_TYPES:
        raise RuntimeError(
            f"Tensor {tensor.name!r} is {tensor.tensor_type}, not a supported "
            f"quantized type ({_QUANTIZED_TYPES})."
        )
    return np.asarray(tensor.data, dtype=np.uint8).tobytes()


def _index_gguf(reader: GGUFReader) -> dict[str, ReaderTensor]:
    """Build a {diffusers_state_dict_key: ReaderTensor} index by stripping the GGUF prefix."""
    index: dict[str, ReaderTensor] = {}
    has_prefix = any(t.name.startswith(GGUF_TENSOR_PREFIX) for t in reader.tensors)
    for t in reader.tensors:
        name = t.name
        if has_prefix and name.startswith(GGUF_TENSOR_PREFIX):
            name = name[len(GGUF_TENSOR_PREFIX):]
        if name in index:
            raise RuntimeError(f"Duplicate tensor key after prefix-stripping: {name!r}")
        index[name] = t
    return index


def _materialize_module_from_meta(module: nn.Module) -> None:
    """Allocate real (zero-initialized) storage for any meta-device parameters/buffers
    held *directly* on this module (not its children). Used right before we overwrite
    those parameters/buffers with values read from the GGUF file."""
    for name, p in list(module._parameters.items()):
        if p is not None and p.is_meta:
            module._parameters[name] = nn.Parameter(
                torch.empty(p.shape, dtype=p.dtype), requires_grad=False
            )
    for name, b in list(module._buffers.items()):
        if b is not None and b.is_meta:
            module._buffers[name] = torch.empty(b.shape, dtype=b.dtype)


def _verify_sha256(path: Path, expected_hex: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_hex:
        raise RuntimeError(
            f"GGUF integrity check failed for {path}: "
            f"expected sha256={expected_hex}, actual={actual}. "
            f"Refusing to load. Re-download or update DECOMPOSER_GGUF_SHA256."
        )


def load_qwen_image_layered_transformer_q8(
    gguf_path: str | Path,
    config_repo: str = "Qwen/Qwen-Image-Layered",
    dtype: torch.dtype = torch.float16,
    expected_sha256: str | None = None,
) -> nn.Module:
    """Build the Qwen-Image-Layered transformer with Q8 GGUF weights.

    See module docstring for the memory contract.
    """
    from diffusers import QwenImageTransformer2DModel

    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        raise FileNotFoundError(gguf_path)

    if expected_sha256:
        _verify_sha256(gguf_path, expected_sha256)

    logger.info("loading GGUF transformer from %s", gguf_path)
    reader = GGUFReader(str(gguf_path))
    gguf_index = _index_gguf(reader)
    logger.info("GGUF indexed: %d tensors", len(gguf_index))

    from accelerate import init_empty_weights

    config = QwenImageTransformer2DModel.load_config(config_repo, subfolder="transformer")
    with init_empty_weights():
        model = QwenImageTransformer2DModel.from_config(config)

    consumed: set[str] = set()

    linear_targets: list[tuple[nn.Module, str, nn.Linear, str]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                full_name = f"{parent_name}.{child_name}" if parent_name else child_name
                linear_targets.append((parent, child_name, child, full_name))

    for parent, child_name, linear, full_name in linear_targets:
        weight_key = f"{full_name}.weight"
        bias_key = f"{full_name}.bias"
        if weight_key not in gguf_index:
            raise RuntimeError(
                f"GGUF file is missing weight for Linear {full_name!r} "
                f"(expected key {weight_key!r}). Cannot proceed without fallback."
            )
        weight_tensor = gguf_index[weight_key]
        out_features, in_features = _logical_shape(weight_tensor)
        if (out_features, in_features) != (linear.out_features, linear.in_features):
            raise RuntimeError(
                f"Shape mismatch for {full_name}: GGUF has {(out_features, in_features)} "
                f"but diffusers expects {(linear.out_features, linear.in_features)}."
            )
        bias_tensor: torch.Tensor | None = None
        if linear.bias is not None:
            if bias_key not in gguf_index:
                raise RuntimeError(
                    f"Linear {full_name!r} has a bias parameter but GGUF is missing "
                    f"{bias_key!r}."
                )
            bias_tensor = _read_dequantized(gguf_index[bias_key])
            consumed.add(bias_key)
        elif bias_key in gguf_index:
            raise RuntimeError(
                f"GGUF has bias {bias_key!r} but diffusers Linear {full_name!r} has none."
            )

        if weight_tensor.tensor_type == GGMLQuantizationType.Q8_0:
            replacement: nn.Module = GgufLinear(
                quantized_weight=_raw_quant_bytes(weight_tensor),
                shape=(out_features, in_features),
                bias=bias_tensor,
                dtype=dtype,
            )
        elif weight_tensor.tensor_type in (GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K):
            quant_name = "Q4_K" if weight_tensor.tensor_type == GGMLQuantizationType.Q4_K else "Q5_K"
            replacement = GgufKQuantLinear(
                quantized_weight=_raw_quant_bytes(weight_tensor),
                shape=(out_features, in_features),
                bias=bias_tensor,
                dtype=dtype,
                quant_type=quant_name,
            )
        else:
            replacement = Bf16Linear(
                weight=_read_dequantized(weight_tensor),
                bias=bias_tensor,
                compute_dtype=dtype,
            )
        setattr(parent, child_name, replacement)
        consumed.add(weight_key)

    state_dict = dict(model.named_parameters(remove_duplicate=False))
    state_dict.update(dict(model.named_buffers(remove_duplicate=False)))

    owners: dict[str, nn.Module] = {name: mod for name, mod in model.named_modules()}

    for full_name, tensor in list(state_dict.items()):
        if full_name in consumed:
            continue
        if full_name not in gguf_index:
            # quants/scales buffers and bias params live inside the swapped-in
            # GgufLinear/Bf16Linear under names that are not present in gguf_index;
            # they were already loaded as part of constructing those modules.
            parent_name = full_name.rsplit(".", 1)[0]
            parent = owners.get(parent_name)
            if isinstance(parent, (GgufLinear, GgufKQuantLinear, Bf16Linear)):
                continue
            raise RuntimeError(
                f"Diffusers parameter/buffer {full_name!r} has no matching tensor in the GGUF file."
            )

        src = _read_dequantized(gguf_index[full_name])
        parent_name, attr = full_name.rsplit(".", 1) if "." in full_name else ("", full_name)
        parent = owners[parent_name]
        _materialize_module_from_meta(parent)
        target = getattr(parent, attr)
        if tuple(src.shape) != tuple(target.shape):
            raise RuntimeError(
                f"Shape mismatch for {full_name!r}: GGUF {tuple(src.shape)} vs "
                f"diffusers {tuple(target.shape)}."
            )
        src = src.to(dtype=target.dtype if target.dtype.is_floating_point else src.dtype)
        with torch.no_grad():
            target.copy_(src)
        consumed.add(full_name)

    # Unconsumed GGUF tensors indicate a name-mapping gap; refusing silent drop
    # preserves the Q8 memory contract (any dropped quantized weight would be
    # silently replaced by a meta-device stub at forward time).
    unconsumed = set(gguf_index.keys()) - consumed
    if unconsumed:
        sample = sorted(unconsumed)[:5]
        raise RuntimeError(
            f"{len(unconsumed)} GGUF tensors were not assigned to any diffusers slot. "
            f"Sample: {sample}. Refusing to silently drop quantized weights."
        )

    del reader, gguf_index
    gc.collect()

    model = model.to(dtype=dtype)
    logger.info("GGUF transformer loaded: %d tensors consumed, dtype=%s", len(consumed), dtype)
    return model
