import types
from pathlib import Path

import numpy as np
import pytest
import torch
from gguf import GGMLQuantizationType
from huggingface_hub import hf_hub_download

from decomposer.core.gguf_loader import GgufLinear
from decomposer.core.gguf_pipeline import (
    _read_dequantized,
    _verify_sha256,
    load_qwen_image_layered_transformer_q8,
)


GGUF_REPO = "unsloth/Qwen-Image-Layered-GGUF"
GGUF_FILE = "qwen-image-layered-Q8_0.gguf"


def test_verify_sha256_rejects_mismatch(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"wrong content")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        _verify_sha256(bad, "0" * 64)


def test_verify_sha256_accepts_match(tmp_path):
    import hashlib

    f = tmp_path / "good.bin"
    f.write_bytes(b"hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    _verify_sha256(f, expected)


def test_read_dequantized_bf16_roundtrip():
    """Bitcast must reinterpret the raw uint8 memmap bytes, not element-cast them."""
    bf16_values = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
    raw_bytes = bf16_values.view(torch.uint8).numpy().reshape(-1)
    fake = types.SimpleNamespace(
        name="x",
        tensor_type=GGMLQuantizationType.BF16,
        shape=np.array([2, 2], dtype=np.uint64),
        data=raw_bytes,
    )
    out = _read_dequantized(fake)
    assert out.dtype == torch.bfloat16
    assert out.shape == (2, 2)
    assert torch.allclose(out.to(torch.float32), bf16_values.to(torch.float32))


def test_read_dequantized_f32_roundtrip():
    f32_values = torch.tensor([1.5, -2.25, 3.0, 4.75], dtype=torch.float32)
    raw_bytes = f32_values.view(torch.uint8).numpy().reshape(-1)
    fake = types.SimpleNamespace(
        name="x",
        tensor_type=GGMLQuantizationType.F32,
        shape=np.array([4], dtype=np.uint64),
        data=raw_bytes,
    )
    out = _read_dequantized(fake)
    assert out.dtype == torch.float32
    assert out.shape == (4,)
    assert torch.equal(out, f32_values)


@pytest.fixture(scope="module")
def gguf_path() -> Path:
    return Path(hf_hub_download(GGUF_REPO, GGUF_FILE))


@pytest.mark.mps_required
def test_loader_returns_transformer_with_gguf_linear_layers(gguf_path):
    model = load_qwen_image_layered_transformer_q8(gguf_path)
    gguf_linear_count = sum(1 for m in model.modules() if isinstance(m, GgufLinear))
    nn_linear_count = sum(
        1
        for m in model.modules()
        if isinstance(m, torch.nn.Linear) and not isinstance(m, GgufLinear)
    )
    assert gguf_linear_count > 50, (
        f"expected many GgufLinear modules, got {gguf_linear_count}"
    )
    assert nn_linear_count == 0, (
        f"no stock nn.Linear should remain, found {nn_linear_count}"
    )


@pytest.mark.mps_required
def test_loader_persistent_memory_is_q8_scale(gguf_path):
    """Total parameter+buffer bytes should be roughly Q8 scale, not fp16 scale."""
    model = load_qwen_image_layered_transformer_q8(gguf_path)
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    assert total_bytes < 28 * 1024**3, (
        f"transformer persistent memory {total_bytes / 1024**3:.1f} GB "
        f"exceeds Q8 budget, Q8 loader is not working as intended"
    )
