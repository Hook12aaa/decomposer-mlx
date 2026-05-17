# MLX DiT Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the Qwen-Image-Layered DiT denoise loop to MLX with fused quantized matmul, targeting 10× per-step speedup (80s → 3-8s) on M3 Max.

**Architecture:** New `MlxBackend` implementing `InferenceBackend`, delegates text-encoder + VAE to existing `MpsBackend`, runs the denoise loop via an MLX-native Qwen transformer adapted from mflux's existing 608-LOC implementation. Offline GGUF → MLX converter handles weight format translation.

**Tech Stack:** Python 3.12, mlx, mlx-nn, mflux (reference only — code adapted, not imported as dependency), existing decomposer infrastructure.

**Key discovery:** mflux (`github.com/filipstrand/mflux`) already has a complete MLX Qwen transformer at `src/mflux/models/qwen/model/qwen_transformer/` — 608 LOC, 10 files, MIT licensed. Uses `mx.fast.scaled_dot_product_attention`, `nn.QuantizedLinear`, and 3D RoPE. We adapt this rather than writing from scratch.

**Spec:** `docs/superpowers/specs/2026-05-20-mlx-dit-port-design.md`

---

## File Map

```
decomposer/
├── core/
│   ├── mlx_backend.py                      # NEW: MlxBackend → InferenceBackend
│   └── backend.py                          # existing (unchanged)
├── mlx_dit/                                # NEW: MLX transformer (adapted from mflux)
│   ├── __init__.py
│   ├── transformer.py                      # QwenTransformer (top-level forward)
│   ├── transformer_block.py                # MMDiTBlock (joint attention + FFN)
│   ├── attention.py                        # Joint attention + RoPE application
│   ├── rope.py                             # 3D RoPE embedding
│   ├── layers.py                           # RMSNorm, LayerNorm, FeedForward, TimestepEmbed
│   └── loader.py                           # load MLX safetensors → populate model
├── mlx_convert/                            # NEW: offline converter
│   ├── __init__.py
│   └── convert.py                          # GGUF → MLX quantized safetensors
├── config.py                               # add: backend, mlx_weights_dir
└── cli.py                                  # add: --backend flag, convert-to-mlx command

tests/
├── test_mlx_dit/
│   ├── __init__.py
│   ├── test_layers.py                      # unit tests for MLX building blocks
│   ├── test_transformer.py                 # forward pass shape test
│   └── test_converter.py                   # GGUF → MLX roundtrip
└── test_mlx_backend.py                     # integration test
```

---

## Task 1: Add MLX dependencies + scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `decomposer/mlx_dit/__init__.py`, `decomposer/mlx_convert/__init__.py`, `tests/test_mlx_dit/__init__.py`
- Modify: `decomposer/config.py`

- [ ] **Step 1: Add mlx deps to pyproject.toml**

Add to dependencies:
```toml
    "mlx>=0.22.0",
    "mlx-nn>=0.22.0",
```

Note: MLX is Apple-Silicon-only. On non-Apple platforms `import mlx` will fail. This is acceptable — MlxBackend is only usable on Apple Silicon.

- [ ] **Step 2: Install**

Run: `uv sync --extra dev`

- [ ] **Step 3: Verify MLX works**

Run: `uv run python -c "import mlx.core as mx; import mlx.nn as nn; print('mlx', mx.__version__); a = mx.ones((2,2)); print(a)"`
Expected: prints mlx version + a 2×2 ones matrix.

- [ ] **Step 4: Create empty packages**

Create `decomposer/mlx_dit/__init__.py`, `decomposer/mlx_convert/__init__.py`, `tests/test_mlx_dit/__init__.py` as empty files.

- [ ] **Step 5: Add Settings fields**

Add to `decomposer/config.py` Settings:
```python
    backend: str = "mps"
    mlx_weights_dir: Path = Path("mlx-weights")
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock decomposer/mlx_dit/ decomposer/mlx_convert/ tests/test_mlx_dit/ decomposer/config.py
git commit -m "Scaffold MLX DiT port: deps + packages + Settings"
```

---

## Task 2: MLX building blocks (layers.py)

**Files:**
- Create: `decomposer/mlx_dit/layers.py`
- Create: `tests/test_mlx_dit/test_layers.py`

Adapt from mflux's `qwen_transformer_rms_norm.py`, `qwen_layer_norm.py`, `qwen_feed_forward.py`, `qwen_timestep_embedding.py`, `qwen_timesteps.py`, `qwen_time_text_embed.py`. Consolidate into one file since they're all small.

- [ ] **Step 1: Write failing tests**

Create `tests/test_mlx_dit/test_layers.py`:
```python
import mlx.core as mx
import mlx.nn as nn
import pytest

from decomposer.mlx_dit.layers import (
    MLXRMSNorm,
    MLXLayerNorm,
    MLXFeedForward,
    MLXTimestepEmbedding,
    MLXTimesteps,
)


def test_rms_norm_preserves_shape():
    norm = MLXRMSNorm(64)
    x = mx.random.normal((2, 10, 64))
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_layer_norm_preserves_shape():
    norm = MLXLayerNorm(64, affine=False)
    x = mx.random.normal((2, 10, 64))
    out = norm(x)
    assert out.shape == (2, 10, 64)


def test_feed_forward_projects():
    ff = MLXFeedForward(dim=64, hidden_dim=256)
    x = mx.random.normal((2, 10, 64))
    out = ff(x)
    assert out.shape == (2, 10, 64)


def test_timestep_embedding_produces_embed():
    te = MLXTimestepEmbedding(in_dim=256, hidden_dim=512, out_dim=64)
    t = mx.random.normal((2, 256))
    out = te(t)
    assert out.shape == (2, 64)


def test_timesteps_produces_sinusoidal():
    ts = MLXTimesteps(num_channels=256)
    t = mx.array([0.1, 0.5])
    out = ts(t)
    assert out.shape == (2, 256)
```

- [ ] **Step 2: Run test to confirm failure**

Run: `uv run pytest tests/test_mlx_dit/test_layers.py -v`
Expected: ModuleNotFoundError

- [ ] **Step 3: Implement `decomposer/mlx_dit/layers.py`**

Adapt from mflux source. Each class is a small `nn.Module`:

```python
import math

import mlx.core as mx
import mlx.nn as nn


class MLXRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rms * self.weight).astype(dtype)


class MLXLayerNorm(nn.Module):
    def __init__(self, dim: int, affine: bool = False, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,)) if affine else None
        self.bias = mx.zeros((dim,)) if affine else None

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        x = (x - mean) * mx.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class MLXFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU(approx="precise")
        self.linear2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.gelu(self.linear1(x)))


class MLXTimesteps(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.num_channels = num_channels

    def __call__(self, t: mx.array) -> mx.array:
        half = self.num_channels // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half) / half)
        args = t[:, None] * freqs[None, :]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class MLXTimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.silu(self.linear1(x)))
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mlx_dit/test_layers.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/mlx_dit/layers.py tests/test_mlx_dit/test_layers.py
git commit -m "MLX layers: RMSNorm, LayerNorm, FeedForward, TimestepEmbedding"
```

---

## Task 3: 3D RoPE in MLX (rope.py)

**Files:**
- Create: `decomposer/mlx_dit/rope.py`
- Create: `tests/test_mlx_dit/test_rope.py`

Adapt from mflux's `qwen_rope.py` (~113 LOC). This is the trickiest piece — 3D rotary embeddings for spatial+temporal position encoding.

- [ ] **Step 1: Write failing test**

Create `tests/test_mlx_dit/test_rope.py`:
```python
import mlx.core as mx
import pytest

from decomposer.mlx_dit.rope import MLXRoPE3D


def test_rope_produces_correct_shapes():
    rope = MLXRoPE3D(axes_dims=[16, 56, 56])
    h, w = 46, 36
    freqs = rope.compute_freqs(height=h, width=w, num_frames=4)
    assert freqs.shape[-1] == 128
    assert freqs.ndim >= 2


def test_rope_apply_preserves_shape():
    rope = MLXRoPE3D(axes_dims=[16, 56, 56])
    q = mx.random.normal((1, 24, 100, 128))
    freqs = rope.compute_freqs(height=10, width=10, num_frames=1)
    out = rope.apply(q, freqs)
    assert out.shape == q.shape
```

- [ ] **Step 2: Confirm fail**

- [ ] **Step 3: Implement `decomposer/mlx_dit/rope.py`**

Adapt from mflux's `qwen_rope.py`. Key structure:
- `compute_freqs(height, width, num_frames)` — precompute rotary frequencies for the 3D grid
- `apply(x, freqs)` — apply rotary embedding via real/imag pair rotation (not complex multiply — MLX doesn't have complex number support, mflux uses cos/sin rotation)

```python
import math

import mlx.core as mx
import mlx.nn as nn


class MLXRoPE3D(nn.Module):
    def __init__(self, axes_dims: list[int], theta: float = 10000.0) -> None:
        super().__init__()
        self.axes_dims = axes_dims
        self.theta = theta

    def _freqs_for_axis(self, dim: int, seq_len: int) -> mx.array:
        freqs = 1.0 / (self.theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        positions = mx.arange(seq_len).astype(mx.float32)
        angles = mx.outer(positions, freqs)
        return mx.concatenate([mx.cos(angles), mx.sin(angles)], axis=-1)

    def compute_freqs(self, height: int, width: int, num_frames: int) -> mx.array:
        t_dim, h_dim, w_dim = self.axes_dims
        t_freqs = self._freqs_for_axis(t_dim, num_frames)
        h_freqs = self._freqs_for_axis(h_dim, height)
        w_freqs = self._freqs_for_axis(w_dim, width)

        t_grid = mx.repeat(t_freqs[:, None, None, :], repeats=height, axis=1)
        t_grid = mx.repeat(t_grid, repeats=width, axis=2)

        h_grid = mx.repeat(h_freqs[None, :, None, :], repeats=num_frames, axis=0)
        h_grid = mx.repeat(h_grid, repeats=width, axis=2)

        w_grid = mx.repeat(w_freqs[None, None, :, :], repeats=num_frames, axis=0)
        w_grid = mx.repeat(w_grid, repeats=height, axis=1)

        freqs = mx.concatenate([t_grid, h_grid, w_grid], axis=-1)
        return freqs.reshape(-1, freqs.shape[-1])

    def apply(self, x: mx.array, freqs: mx.array) -> mx.array:
        seq_len = x.shape[-2]
        freqs = freqs[:seq_len]
        half = freqs.shape[-1] // 2
        cos_f = freqs[:, :half]
        sin_f = freqs[:, half:]
        x1 = x[..., :half]
        x2 = x[..., half:]
        out1 = x1 * cos_f - x2 * sin_f
        out2 = x2 * cos_f + x1 * sin_f
        return mx.concatenate([out1, out2], axis=-1)
```

Note: this is a simplified version. The actual mflux implementation may handle the grid construction differently — the subagent implementing this task should read the real mflux source at `https://github.com/filipstrand/mflux/blob/main/src/mflux/models/qwen/model/qwen_transformer/qwen_rope.py` and adapt as needed. The test verifies shape correctness; numerical equivalence with the PyTorch version is validated in Task 7.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mlx_dit/test_rope.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/mlx_dit/rope.py tests/test_mlx_dit/test_rope.py
git commit -m "MLX 3D RoPE embedding for spatial+temporal position encoding"
```

---

## Task 4: Joint attention block (attention.py + transformer_block.py)

**Files:**
- Create: `decomposer/mlx_dit/attention.py`
- Create: `decomposer/mlx_dit/transformer_block.py`
- Create: `tests/test_mlx_dit/test_attention.py`

Adapt from mflux's `qwen_attention.py` (~158 LOC) and `qwen_transformer_block.py` (~76 LOC).

- [ ] **Step 1: Write failing test**

Create `tests/test_mlx_dit/test_attention.py`:
```python
import mlx.core as mx
import pytest

from decomposer.mlx_dit.attention import MLXJointAttention
from decomposer.mlx_dit.transformer_block import MLXMMDiTBlock


def test_joint_attention_produces_correct_shapes():
    attn = MLXJointAttention(dim=64, num_heads=4, head_dim=16)
    img = mx.random.normal((1, 20, 64))
    txt = mx.random.normal((1, 10, 64))
    freqs = mx.random.normal((30, 32))
    img_out, txt_out = attn(img, txt, freqs)
    assert img_out.shape == (1, 20, 64)
    assert txt_out.shape == (1, 10, 64)


def test_mmdit_block_produces_correct_shapes():
    block = MLXMMDiTBlock(dim=64, num_heads=4, head_dim=16, mlp_ratio=4.0)
    img = mx.random.normal((1, 20, 64))
    txt = mx.random.normal((1, 10, 64))
    mod = mx.random.normal((1, 64))
    freqs = mx.random.normal((30, 32))
    img_out, txt_out = block(img, txt, mod, freqs)
    assert img_out.shape == (1, 20, 64)
    assert txt_out.shape == (1, 10, 64)
```

- [ ] **Step 2: Confirm fail**

- [ ] **Step 3: Implement**

`decomposer/mlx_dit/attention.py` — joint attention with RoPE, using `mx.fast.scaled_dot_product_attention`:

```python
import mlx.core as mx
import mlx.nn as nn

from decomposer.mlx_dit.layers import MLXRMSNorm
from decomposer.mlx_dit.rope import MLXRoPE3D


class MLXJointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.to_q_img = nn.Linear(dim, num_heads * head_dim)
        self.to_k_img = nn.Linear(dim, num_heads * head_dim)
        self.to_v_img = nn.Linear(dim, num_heads * head_dim)
        self.to_q_txt = nn.Linear(dim, num_heads * head_dim)
        self.to_k_txt = nn.Linear(dim, num_heads * head_dim)
        self.to_v_txt = nn.Linear(dim, num_heads * head_dim)

        self.norm_q_img = MLXRMSNorm(head_dim)
        self.norm_k_img = MLXRMSNorm(head_dim)
        self.norm_q_txt = MLXRMSNorm(head_dim)
        self.norm_k_txt = MLXRMSNorm(head_dim)

        self.to_out_img = [nn.Linear(num_heads * head_dim, dim)]
        self.to_out_txt = [nn.Linear(num_heads * head_dim, dim)]

        self.rope = MLXRoPE3D(axes_dims=[16, 56, 56])

    def __call__(self, img: mx.array, txt: mx.array, freqs: mx.array) -> tuple[mx.array, mx.array]:
        B, S_img, _ = img.shape
        _, S_txt, _ = txt.shape

        q_img = self._reshape_heads(self.to_q_img(img))
        k_img = self._reshape_heads(self.to_k_img(img))
        v_img = self._reshape_heads(self.to_v_img(img))
        q_txt = self._reshape_heads(self.to_q_txt(txt))
        k_txt = self._reshape_heads(self.to_k_txt(txt))
        v_txt = self._reshape_heads(self.to_v_txt(txt))

        q_img = self.norm_q_img(q_img)
        k_img = self.norm_k_img(k_img)
        q_txt = self.norm_q_txt(q_txt)
        k_txt = self.norm_k_txt(k_txt)

        q_img = self.rope.apply(q_img, freqs)
        k_img = self.rope.apply(k_img, freqs)

        q = mx.concatenate([txt_q := q_txt, q_img], axis=2)
        k = mx.concatenate([k_txt, k_img], axis=2)
        v = mx.concatenate([v_txt, v_img], axis=2)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

        txt_out = out[:, :, :S_txt, :]
        img_out = out[:, :, S_txt:, :]

        txt_out = txt_out.transpose(0, 2, 1, 3).reshape(B, S_txt, -1)
        img_out = img_out.transpose(0, 2, 1, 3).reshape(B, S_img, -1)

        img_out = self.to_out_img[0](img_out)
        txt_out = self.to_out_txt[0](txt_out)

        return img_out, txt_out

    def _reshape_heads(self, x: mx.array) -> mx.array:
        B, S, _ = x.shape
        return x.reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
```

`decomposer/mlx_dit/transformer_block.py` — single MMDiT block with modulation:

```python
import mlx.core as mx
import mlx.nn as nn

from decomposer.mlx_dit.attention import MLXJointAttention
from decomposer.mlx_dit.layers import MLXFeedForward, MLXLayerNorm


class MLXMMDiTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        mlp_dim = int(dim * mlp_ratio)

        self.img_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.img_norm1 = MLXLayerNorm(dim, affine=False)
        self.img_norm2 = MLXLayerNorm(dim, affine=False)
        self.img_mlp = MLXFeedForward(dim, mlp_dim)

        self.txt_mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        self.txt_norm1 = MLXLayerNorm(dim, affine=False)
        self.txt_norm2 = MLXLayerNorm(dim, affine=False)
        self.txt_mlp = MLXFeedForward(dim, mlp_dim)

        self.attn = MLXJointAttention(dim, num_heads, head_dim)

    def __call__(self, img: mx.array, txt: mx.array, mod: mx.array, freqs: mx.array) -> tuple[mx.array, mx.array]:
        img_mod = self.img_mod(mod).reshape(1, 6, -1)
        img_shift1, img_scale1, img_gate1 = img_mod[:, 0], img_mod[:, 1], img_mod[:, 2]
        img_shift2, img_scale2, img_gate2 = img_mod[:, 3], img_mod[:, 4], img_mod[:, 5]

        txt_mod_out = self.txt_mod(mod).reshape(1, 6, -1)
        txt_shift1, txt_scale1, txt_gate1 = txt_mod_out[:, 0], txt_mod_out[:, 1], txt_mod_out[:, 2]
        txt_shift2, txt_scale2, txt_gate2 = txt_mod_out[:, 3], txt_mod_out[:, 4], txt_mod_out[:, 5]

        img_norm = self.img_norm1(img) * (1 + img_scale1[:, None, :]) + img_shift1[:, None, :]
        txt_norm = self.txt_norm1(txt) * (1 + txt_scale1[:, None, :]) + txt_shift1[:, None, :]

        img_attn, txt_attn = self.attn(img_norm, txt_norm, freqs)
        img = img + img_gate1[:, None, :] * img_attn
        txt = txt + txt_gate1[:, None, :] * txt_attn

        img_norm2 = self.img_norm2(img) * (1 + img_scale2[:, None, :]) + img_shift2[:, None, :]
        txt_norm2 = self.txt_norm2(txt) * (1 + txt_scale2[:, None, :]) + txt_shift2[:, None, :]

        img = img + img_gate2[:, None, :] * self.img_mlp(img_norm2)
        txt = txt + txt_gate2[:, None, :] * self.txt_mlp(txt_norm2)

        return img, txt
```

Note: the `axes_dims` for RoPE should be read from the config in the full transformer, not hardcoded in attention. The subagent should make this configurable. The code above is a starting point.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mlx_dit/test_attention.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add decomposer/mlx_dit/attention.py decomposer/mlx_dit/transformer_block.py tests/test_mlx_dit/test_attention.py
git commit -m "MLX joint attention + MMDiT block with modulation"
```

---

## Task 5: Full transformer (transformer.py)

**Files:**
- Create: `decomposer/mlx_dit/transformer.py`
- Modify: `tests/test_mlx_dit/test_attention.py` (append test)

Adapt from mflux's `qwen_transformer.py` (~139 LOC). Stacks 60 blocks, adds embeddings, norm_out, proj_out.

- [ ] **Step 1: Append failing test**

Append to `tests/test_mlx_dit/test_attention.py` (or create `tests/test_mlx_dit/test_transformer.py`):
```python
from decomposer.mlx_dit.transformer import MLXQwenTransformer


def test_transformer_forward_produces_output():
    config = {
        "num_layers": 2,
        "num_attention_heads": 4,
        "attention_head_dim": 16,
        "in_channels": 64,
        "out_channels": 16,
        "joint_attention_dim": 128,
        "axes_dims_rope": [16, 56, 56],
    }
    model = MLXQwenTransformer(config)
    latent = mx.random.normal((1, 20, 64))
    cond = mx.random.normal((1, 10, 128))
    timestep = mx.array([0.5])
    out = model(latent, cond, timestep, height=4, width=5, num_frames=1)
    assert out.shape[0] == 1
    assert out.shape[-1] == 16
```

- [ ] **Step 2: Confirm fail**

- [ ] **Step 3: Implement `decomposer/mlx_dit/transformer.py`**

The transformer orchestrates: patch embed → stack blocks → norm → project out.

```python
import mlx.core as mx
import mlx.nn as nn

from decomposer.mlx_dit.layers import (
    MLXFeedForward,
    MLXLayerNorm,
    MLXRMSNorm,
    MLXTimestepEmbedding,
    MLXTimesteps,
)
from decomposer.mlx_dit.rope import MLXRoPE3D
from decomposer.mlx_dit.transformer_block import MLXMMDiTBlock


class MLXQwenTransformer(nn.Module):
    def __init__(self, config: dict) -> None:
        super().__init__()
        num_layers = config["num_layers"]
        num_heads = config["num_attention_heads"]
        head_dim = config["attention_head_dim"]
        dim = num_heads * head_dim
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        joint_dim = config["joint_attention_dim"]
        axes_dims = config["axes_dims_rope"]
        patch_size = config.get("patch_size", 2)

        self.dim = dim
        self.out_channels = out_channels
        self.patch_size = patch_size

        self.img_in = nn.Linear(in_channels, dim)
        self.txt_in = nn.Linear(joint_dim, dim)
        self.txt_norm = MLXRMSNorm(joint_dim)

        self.time_embed = MLXTimesteps(num_channels=256)
        self.time_proj = MLXTimestepEmbedding(in_dim=256, hidden_dim=dim, out_dim=dim)

        self.rope = MLXRoPE3D(axes_dims=axes_dims)

        self.blocks = [
            MLXMMDiTBlock(dim=dim, num_heads=num_heads, head_dim=head_dim)
            for _ in range(num_layers)
        ]

        self.norm_out_linear = nn.Linear(dim, dim)
        self.norm_out_norm = MLXLayerNorm(dim, affine=False)
        self.proj_out = nn.Linear(dim, patch_size * patch_size * out_channels)

    def __call__(
        self,
        latent: mx.array,
        cond: mx.array,
        timestep: mx.array,
        *,
        height: int,
        width: int,
        num_frames: int,
    ) -> mx.array:
        t_emb = self.time_proj(self.time_embed(timestep))
        img = self.img_in(latent)
        txt = self.txt_in(self.txt_norm(cond))

        freqs = self.rope.compute_freqs(height=height, width=width, num_frames=num_frames)

        for block in self.blocks:
            img, txt = block(img, txt, t_emb, freqs)

        shift, scale = self.norm_out_linear(nn.silu(t_emb)).chunk(2, axis=-1)
        img = self.norm_out_norm(img) * (1 + scale[:, None, :]) + shift[:, None, :]
        img = self.proj_out(img)

        return img
```

Note: the actual mflux implementation may handle `addition_t_cond`, packing/unpacking, and the VLD head differently. The subagent MUST read the real mflux Qwen transformer source and adapt. This code is a structural template.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_mlx_dit/ -v`
Expected: all tests pass (7+ total across layers, rope, attention, transformer).

- [ ] **Step 5: Commit**

```bash
git add decomposer/mlx_dit/transformer.py tests/test_mlx_dit/
git commit -m "MLX Qwen transformer: stack 60 blocks + embeddings + output projection"
```

---

## Task 6: GGUF → MLX weight converter

**Files:**
- Create: `decomposer/mlx_convert/convert.py`
- Create: `tests/test_mlx_dit/test_converter.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_mlx_dit/test_converter.py`:
```python
from pathlib import Path

import mlx.core as mx
import pytest

from decomposer.mlx_convert.convert import build_name_mapping, convert_gguf_to_mlx


def test_build_name_mapping_maps_known_keys():
    mapping = build_name_mapping()
    assert "img_in.weight" in mapping
    assert "transformer_blocks.0.attn.to_q.weight" in mapping or "blocks.0.attn.to_q_img.weight" in mapping.values()


@pytest.mark.mps_required
def test_convert_produces_safetensors(tmp_path):
    from huggingface_hub import hf_hub_download
    gguf_path = hf_hub_download("unsloth/Qwen-Image-Layered-GGUF", "qwen-image-layered-Q8_0.gguf")
    out_dir = tmp_path / "mlx-out"
    convert_gguf_to_mlx(gguf_path, out_dir, bits=4)
    assert (out_dir / "config.json").exists()
    safetensors_files = list(out_dir.glob("*.safetensors"))
    assert len(safetensors_files) >= 1
```

- [ ] **Step 2: Confirm fail**

- [ ] **Step 3: Implement `decomposer/mlx_convert/convert.py`**

```python
from __future__ import annotations

import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from gguf import GGMLQuantizationType, GGUFReader

logger = logging.getLogger(__name__)


def build_name_mapping() -> dict[str, str]:
    mapping = {}
    mapping["img_in.weight"] = "img_in.weight"
    mapping["img_in.bias"] = "img_in.bias"
    mapping["txt_in.weight"] = "txt_in.weight"
    mapping["txt_in.bias"] = "txt_in.bias"
    mapping["txt_norm.weight"] = "txt_norm.weight"
    mapping["norm_out.linear.weight"] = "norm_out_linear.weight"
    mapping["norm_out.linear.bias"] = "norm_out_linear.bias"
    mapping["proj_out.weight"] = "proj_out.weight"
    mapping["proj_out.bias"] = "proj_out.bias"
    mapping["time_text_embed.timestep_embedder.linear_1.weight"] = "time_proj.linear1.weight"
    mapping["time_text_embed.timestep_embedder.linear_1.bias"] = "time_proj.linear1.bias"
    mapping["time_text_embed.timestep_embedder.linear_2.weight"] = "time_proj.linear2.weight"
    mapping["time_text_embed.timestep_embedder.linear_2.bias"] = "time_proj.linear2.bias"

    for i in range(60):
        prefix = f"transformer_blocks.{i}"
        mlx_prefix = f"blocks.{i}"
        for side in ["img", "txt"]:
            mapping[f"{prefix}.{side}_mod.lin.weight"] = f"{mlx_prefix}.{side}_mod.layers.1.weight"
            mapping[f"{prefix}.{side}_mod.lin.bias"] = f"{mlx_prefix}.{side}_mod.layers.1.bias"
            for proj in ["to_q", "to_k", "to_v"]:
                mapping[f"{prefix}.attn.{proj}.weight"] = f"{mlx_prefix}.attn.{proj}_{side}.weight"
            mapping[f"{prefix}.attn.to_out.0.weight"] = f"{mlx_prefix}.attn.to_out_{side}.0.weight"
            mapping[f"{prefix}.attn.norm_{proj[3]}.weight"] = f"{mlx_prefix}.attn.norm_{proj[3]}_{side}.weight"
            for lyr in ["net.0.proj", "net.2"]:
                k = f"{prefix}.{side}_mlp.{lyr}"
                mlx_k = k.replace(prefix, mlx_prefix).replace("net.0.proj", "linear1").replace("net.2", "linear2")
                mapping[f"{k}.weight"] = f"{mlx_k}.weight"
                mapping[f"{k}.bias"] = f"{mlx_k}.bias"
    return mapping


def _read_tensor_as_fp32(tensor) -> np.ndarray:
    raw = tensor.data.tobytes()
    shape = tuple(int(x) for x in reversed(tensor.shape.tolist()))
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
        n = int(np.prod(shape))
        packed = np.asarray(tensor.data, dtype=np.uint8).tobytes()
        q, s = _unpack_q8_0_to_tensors(packed, shape)
        return (q.float() * s.float().unsqueeze(-1)).view(*shape).numpy()
    raise RuntimeError(f"unsupported tensor type {t} for {tensor.name}")


def convert_gguf_to_mlx(
    gguf_path: str | Path,
    output_dir: Path,
    bits: int = 4,
    group_size: int = 64,
    config_repo: str = "Qwen/Qwen-Image-Layered",
) -> None:
    from diffusers import QwenImageTransformer2DModel

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = dict(QwenImageTransformer2DModel.load_config(config_repo, subfolder="transformer"))
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    logger.info("config written")

    reader = GGUFReader(str(gguf_path))
    gguf_index = {}
    for t in reader.tensors:
        gguf_index[t.name] = t
    logger.info("GGUF indexed: %d tensors", len(gguf_index))

    name_map = build_name_mapping()
    weights = {}

    for gguf_name, tensor in gguf_index.items():
        mlx_name = name_map.get(gguf_name)
        if mlx_name is None:
            logger.warning("unmapped GGUF tensor: %s (skipped)", gguf_name)
            continue
        arr = _read_tensor_as_fp32(tensor)
        mlx_arr = mx.array(arr)

        if "weight" in mlx_name and arr.ndim == 2 and arr.shape[0] >= 64 and arr.shape[1] >= 64:
            q_weight, q_scales, q_biases = mx.quantize(mlx_arr, group_size=group_size, bits=bits)
            weights[mlx_name] = q_weight
            weights[mlx_name.replace(".weight", ".scales")] = q_scales
            weights[mlx_name.replace(".weight", ".biases")] = q_biases
        else:
            weights[mlx_name] = mlx_arr.astype(mx.float16)
        logger.info("converted %s -> %s", gguf_name, mlx_name)

    mx.save_safetensors(str(output_dir / "weights.safetensors"), weights)
    logger.info("saved %d tensors to %s", len(weights), output_dir)
```

Note: The name mapping above is approximate. The subagent MUST verify the exact mapping by inspecting both the GGUF tensor names (via `reader.tensors`) and the MLX model parameter names (via `model.parameters()`) — compare the two lists and fix any discrepancies. mflux's weight loading code is the reference for how they solved the same problem.

- [ ] **Step 4: Run unit test (not the MPS-required integration test)**

Run: `uv run pytest tests/test_mlx_dit/test_converter.py::test_build_name_mapping_maps_known_keys -v`
Expected: 1 passed.

The integration test (`test_convert_produces_safetensors`) is MPS-required and slow (~10 min for the dequant+requant of 20 GB). Run it separately:

Run: `uv run pytest tests/test_mlx_dit/test_converter.py -v -m mps_required` (optional, slow)

- [ ] **Step 5: Commit**

```bash
git add decomposer/mlx_convert/convert.py tests/test_mlx_dit/test_converter.py
git commit -m "GGUF to MLX converter: dequant + re-quantize + name mapping"
```

---

## Task 7: Weight loader (loader.py)

**Files:**
- Create: `decomposer/mlx_dit/loader.py`

Load the converted MLX safetensors into the MLX transformer, then apply `nn.quantize` to replace `nn.Linear` with `nn.QuantizedLinear`.

- [ ] **Step 1: Implement `decomposer/mlx_dit/loader.py`**

```python
import json
import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from decomposer.mlx_dit.transformer import MLXQwenTransformer

logger = logging.getLogger(__name__)


def load_mlx_transformer(weights_dir: Path, bits: int = 4, group_size: int = 64) -> MLXQwenTransformer:
    config = json.loads((weights_dir / "config.json").read_text())
    model = MLXQwenTransformer(config)

    nn.quantize(model, group_size=group_size, bits=bits)

    weights = mx.load(str(weights_dir / "weights.safetensors"))
    model.load_weights(list(weights.items()))

    mx.eval(model.parameters())
    logger.info("MLX transformer loaded: %d parameters, %d-bit quantized", len(weights), bits)
    return model
```

- [ ] **Step 2: Commit**

```bash
git add decomposer/mlx_dit/loader.py
git commit -m "MLX transformer loader: load quantized safetensors into model"
```

---

## Task 8: MlxBackend + integration test

**Files:**
- Create: `decomposer/core/mlx_backend.py`
- Create: `tests/test_mlx_backend.py`
- Modify: `decomposer/cli.py` (add `--backend` flag)
- Modify: `decomposer/web/app.py` (backend selection from Settings)

- [ ] **Step 1: Implement `decomposer/core/mlx_backend.py`**

```python
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from decomposer.config import Settings, get_settings
from decomposer.core.mps_backend import MpsBackend
from decomposer.core.xray import Tracer

logger = logging.getLogger(__name__)


class MlxBackend:
    def __init__(self, settings: Settings | None = None, device: str = "mps") -> None:
        self.settings = settings if settings is not None else get_settings()
        self._mps = MpsBackend(settings=self.settings, device=device)

    def decompose(
        self,
        image: Image.Image,
        layers: int,
        resolution: int = 640,
        steps: int = 8,
        seed: int | None = None,
        tracer: Tracer | None = None,
    ) -> list[Image.Image]:
        import mlx.core as mx

        t = tracer or Tracer(run_id="adhoc")
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        logger.info("mlx decompose start run_id=%s layers=%d steps=%d", t.run_id, layers, steps)

        cond = self._mps._encode_prompt(image, prompt="marketing asset", tracer=t)
        image_latent, height, width = self._mps._encode_image_to_latent(
            image, resolution=resolution, tracer=t
        )

        with t.stage("load_dit"):
            from decomposer.mlx_dit.loader import load_mlx_transformer
            model = load_mlx_transformer(self.settings.mlx_weights_dir)

        cond_mlx = mx.array(cond.float().numpy())
        latent_mlx = mx.array(image_latent.float().numpy())

        with t.stage("denoise_loop", steps=steps, layers=layers, resolution=resolution):
            from decomposer.mlx_dit.scheduler import flow_match_euler_step, get_sigmas
            sigmas = get_sigmas(steps)
            noise = mx.random.normal(latent_mlx.shape)
            latent_noised = latent_mlx * sigmas[0] + noise * (1 - sigmas[0])

            h_lat = height // (model.patch_size * 2)
            w_lat = width // (model.patch_size * 2)

            for i in range(steps):
                with t.step("denoise_step", i=i):
                    pred = model(
                        latent_noised, cond_mlx, mx.array([sigmas[i]]),
                        height=h_lat, width=w_lat, num_frames=layers + 1,
                    )
                    mx.eval(pred)
                    latent_noised = flow_match_euler_step(latent_noised, pred, sigmas[i], sigmas[i + 1])
                    mx.eval(latent_noised)

        with t.stage("free_dit"):
            del model
            import gc
            gc.collect()

        latents_torch = torch.from_numpy(np.array(latent_noised))
        return self._mps._decode(latents_torch, layers=layers, height=height, width=width, tracer=t)
```

Note: This is a structural template. The exact scheduler logic, latent packing, and h/w computation need to match the Qwen-Image-Layered pipeline's internal behavior. The subagent implementing this MUST read mflux's inference loop and the diffusers pipeline's `__call__` to get the details right. `scheduler.py` (referenced below) needs to be created as a small helper module.

- [ ] **Step 2: Create `decomposer/mlx_dit/scheduler.py`**

```python
import mlx.core as mx


def get_sigmas(num_steps: int) -> list[float]:
    sigmas = [(1.0 - i / num_steps) for i in range(num_steps + 1)]
    return sigmas


def flow_match_euler_step(
    latent: mx.array, pred: mx.array, sigma: float, sigma_next: float
) -> mx.array:
    dt = sigma_next - sigma
    return latent + pred * dt
```

- [ ] **Step 3: Add `--backend` flag to CLI**

Modify `decomposer/cli.py` `decompose` command to accept `--backend` option:
```python
backend_name: str = typer.Option("mps", help="Backend: mps, mlx, or fake"),
```

In the command body, select the backend:
```python
if backend_name == "mlx":
    from decomposer.core.mlx_backend import MlxBackend
    backend = MlxBackend(settings=settings)
elif backend_name == "fake":
    backend = FakeBackend(latency_ms=50)
else:
    from decomposer.core.mps_backend import MpsBackend
    backend = MpsBackend(settings=settings)
```

- [ ] **Step 4: Add `convert-to-mlx` CLI command**

Add to `decomposer/cli.py`:
```python
@app.command("convert-to-mlx")
def convert_to_mlx(
    gguf: Annotated[Path, typer.Argument(exists=True)],
    output: Path = Path("mlx-weights"),
    bits: int = 4,
) -> None:
    """Convert GGUF weights to MLX quantized safetensors."""
    from decomposer.mlx_convert.convert import convert_gguf_to_mlx
    console.print(f"Converting {gguf} -> {output} ({bits}-bit)")
    convert_gguf_to_mlx(gguf, output, bits=bits)
    console.print(f"[green]Done. MLX weights at {output}[/green]")
```

- [ ] **Step 5: Write integration test**

Create `tests/test_mlx_backend.py`:
```python
import pytest
from PIL import Image

from decomposer.core.xray import Tracer


@pytest.mark.mps_required
def test_mlx_backend_decompose_produces_layers():
    from decomposer.core.mlx_backend import MlxBackend
    backend = MlxBackend()
    img = Image.new("RGB", (64, 64), (200, 50, 50))
    t = Tracer(run_id="r-mlx")
    layers = backend.decompose(img, layers=3, resolution=640, steps=4, tracer=t)
    assert len(layers) == 3
    assert all(layer.mode == "RGBA" for layer in layers)
    rep = t.report()
    seen = [s.name for s in rep.stages]
    for expected in ["load_text_encoder", "load_dit", "denoise_loop", "decode_layers"]:
        assert expected in seen, f"missing stage: {expected}"
```

- [ ] **Step 6: Commit**

```bash
git add decomposer/core/mlx_backend.py decomposer/mlx_dit/scheduler.py decomposer/cli.py tests/test_mlx_backend.py
git commit -m "MlxBackend: MLX denoise loop with MpsBackend delegation for text encoder + VAE"
```

---

## Done criteria

- `decomposer convert-to-mlx <gguf-path> --output mlx-weights/ --bits 4` produces valid MLX safetensors
- `decomposer decompose test_image.jpg --backend mlx --layers 3 --steps 4` produces non-degenerate RGBA layers
- X-ray trace shows `denoise_loop.wall_ms` < 100s for 8 steps (target: < 12.5s/step)
- All unit tests pass in CI (non-MPS tests for layers/rope/attention shapes)
- Integration test passes on M3 Max

## Out of scope (deferred)

- Performance tuning (`mx.compile`, memory optimization) — separate follow-up after correctness verified
- Text encoder MLX port — MpsBackend delegation is fast enough
- VAE MLX port — same
- Auto-researcher experiment for MLX vs MPS comparison — run after integration works
