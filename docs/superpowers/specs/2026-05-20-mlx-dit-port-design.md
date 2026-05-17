# MLX DiT Port — Design Spec

**Date:** 2026-05-20
**Status:** Approved for implementation planning
**Project name:** `decomposer.mlx_dit` (subpackage of existing decomposer project)

## Goal

Port the Qwen-Image-Layered DiT (20B-param MMDiT transformer) to Apple's MLX framework with native fused quantized matmul kernels. Target: **10× per-step speedup** (current ~80s/step on M3 Max → ~3-8s/step) by eliminating the Python-level per-call dequant bottleneck in GgufLinear.

The research (2026-05-18) showed the system is 1000× off the M3 Max bandwidth ceiling — the bottleneck is not data volume but GgufLinear's eager `(quants * scales).view(*shape)` dequant on every forward call. MLX's `mlx.core.quantized_matmul` performs dequant inside the Metal kernel, eliminating the overhead.

## Non-goals

- Porting the text encoder (Qwen2.5-VL-7B) to MLX — stays on PyTorch MPS. Fast enough (~45s, <5% of wall time). Phase 2 if needed.
- Porting the VAE to MLX — stays on PyTorch MPS. ~7s total. Not worth the port.
- Replacing MpsBackend — stays as fallback. MlxBackend is a new backend alongside it.
- Training or fine-tuning in MLX — inference only.
- Supporting non-Apple hardware — MLX is Apple Silicon only by design.

## Critical insight that shapes this design

From the 2026-05-18 optimization research:

> "M3 Max bandwidth: ~300 GB/s sustained. 21 GB Q8 DiT → bandwidth floor per step: ~70 ms. We observe: 77,000 ms per step. We're ~1000× off the bandwidth ceiling. The per-call Python dequant in GgufLinear.forward is the bottleneck."

MLX solves this by fusing dequant into the matmul Metal kernel. The weight data never materializes as a full-precision tensor — the Metal shader reads packed quantized bytes and dequants per-element during the matmul. Zero Python-level dequant overhead.

## Architecture

```
decomposer/
├── core/
│   ├── mps_backend.py              # existing (unchanged)
│   ├── mlx_backend.py              # NEW: MlxBackend → InferenceBackend
│   └── backend.py                  # existing Protocol (unchanged)
├── mlx_dit/                        # NEW: pure MLX transformer
│   ├── __init__.py
│   ├── transformer.py              # QwenImageLayeredTransformer
│   ├── blocks.py                   # MMDiTBlock (joint attention + FFN)
│   ├── attention.py                # MultiHeadAttention + 3D RoPE
│   ├── embeddings.py               # TimestepEmbed, PatchEmbed, TextProjection
│   ├── layers.py                   # RMSNorm, MLP, AdaLayerNorm, modulation
│   └── loader.py                   # load weights from MLX safetensors
├── mlx_convert/                    # NEW: offline GGUF → MLX converter
│   ├── __init__.py
│   └── convert.py                  # GGUF → MLX quantized safetensors
└── config.py                       # add: backend, mlx_weights_dir
```

### MlxBackend flow

```
MlxBackend.decompose(image, layers, resolution, steps):
  1. _encode_prompt          → delegate to MpsBackend (PyTorch MPS)
  2. _encode_image_to_latent → delegate to MpsBackend (PyTorch MPS)
  3. _denoise                → PURE MLX:
     a. Load MLX transformer from mlx_weights_dir (quantized safetensors)
     b. Convert conditioning tensors: torch → numpy → mlx.core.array
     c. FlowMatch scheduler loop (reimplemented in MLX/numpy):
        for t in timesteps:
          latent = mlx_transformer(latent, cond, t)  # fused quantized matmul
          latent = scheduler_step(latent, noise_pred, t)
     d. Convert result: mlx.core.array → numpy → torch.Tensor
  4. _decode                 → delegate to MpsBackend (PyTorch MPS)
```

The PyTorch↔MLX bridge is two numpy array transfers per inference (at phase 2→3 and 3→4 boundaries). Cost: microseconds vs. the seconds-scale inference.

### Backend selection

Settings field: `backend: str = "mps"` (options: `"mps"`, `"mlx"`, `"fake"`).

CLI: `decomposer decompose ... --backend mlx`
Web: `DECOMPOSER_BACKEND=mlx`

`create_app` and `cli.py` read `settings.backend` and instantiate the right backend class.

## MLX transformer architecture

Based on diffusers' `QwenImageTransformer2DModel` (60 transformer blocks, ~20B params):

```
QwenImageLayeredTransformer(mlx.nn.Module)
├── img_in: QuantizedLinear           # image latent → patch tokens
├── txt_in: QuantizedLinear           # text hidden states → model dim
├── time_embed: TimestepEmbedding     # timestep → modulation vector
│   └── MLP (Linear → SiLU → Linear)
├── blocks: 60 × MMDiTBlock
│   ├── norm1_img / norm1_txt: AdaLayerNorm (modulated by timestep)
│   ├── attn: JointAttention
│   │   ├── to_q_img / to_k_img / to_v_img: QuantizedLinear
│   │   ├── to_q_txt / to_k_txt / to_v_txt: QuantizedLinear
│   │   ├── rope_3d: QwenEmbedLayer3DRope (precomputed frequencies)
│   │   ├── sdpa: mx.fast.scaled_dot_product_attention (MLX fused op)
│   │   └── to_out_img / to_out_txt: QuantizedLinear
│   ├── norm2_img / norm2_txt: AdaLayerNorm
│   └── ffn_img / ffn_txt: MLP
│       └── gate_proj + up_proj → SiLU → down_proj (QuantizedLinear)
├── norm_out: AdaLayerNorm
├── proj_out: QuantizedLinear          # → output latent
└── (VLD conditioning is applied via the input latent packing, not a separate head)
```

All `QuantizedLinear` modules use `mlx.nn.QuantizedLinear(group_size=64, bits=4)` (or bits=8). Weights stay packed in Metal buffer memory. Dequant happens inside the matmul kernel.

`mx.fast.scaled_dot_product_attention` is MLX's fused attention op — handles the Q/K/V → softmax → V multiply in a single Metal dispatch. Combined with quantized Q/K/V projections, this is where the 10-30× speedup lives.

## Weight conversion pipeline

Offline, run once:

```bash
decomposer convert-to-mlx \
  --gguf ~/.cache/huggingface/.../qwen-image-layered-Q8_0.gguf \
  --config Qwen/Qwen-Image-Layered \
  --output mlx-weights/ \
  --bits 4
```

Steps:
1. Read GGUF tensors via existing `_index_gguf` + `_read_dequantized` (already implemented)
2. Map GGUF tensor names → MLX model parameter names (naming convention differs)
3. For each weight tensor:
   - Dequantize to fp32 (transient, one tensor at a time — bounded peak memory)
   - Re-quantize using `mlx.core.quantize(weight, group_size=64, bits=bits)`
   - Store quantized weight + scales + biases
4. Save as MLX safetensors to `mlx-weights/`
5. Write `mlx-weights/config.json` with architecture config (num_layers=60, hidden_dim, num_heads, etc.)

For non-quantized tensors (norms, embeddings, biases): store as fp16 directly.

Output: `mlx-weights/` directory (~10 GB for 4-bit, ~20 GB for 8-bit).

### Name mapping

GGUF uses diffusers-style names: `transformer_blocks.0.attn.to_q.weight`
MLX model uses our names: `blocks.0.attn.to_q_img.weight`

The mapping is a deterministic rename dictionary built by inspecting both naming conventions. Stored in `mlx_dit/loader.py`.

## Scheduler

The FlowMatch Euler scheduler is simple enough to reimplement in pure numpy/MLX (~50 lines). It computes:
- `sigma` schedule from the number of steps
- Per-step: `latent = latent + (noise_pred - latent) * dt`

No PyTorch dependency needed. The scheduler state (sigma schedule, timesteps) is computed once at the start of the denoise loop.

## Testing strategy

1. **Per-layer numerical equivalence**: for a single MMDiTBlock, load the same fp32 weights into both PyTorch and MLX, feed the same input, compare outputs. Must match within `atol=1e-4` for fp32, `atol=1e-2` for quantized.

2. **Weight converter round-trip**: GGUF → MLX → load → verify parameter shapes and dtypes match the architecture config.

3. **End-to-end smoke test**: `MlxBackend.decompose(test_image)` produces non-degenerate RGBA layers. Use the auto-researcher's oracle to compare vs MpsBackend baseline (structural similarity, not pixel-perfect).

4. **Performance benchmark**: automated via the X-ray tracer. `denoise_loop.wall_ms` must be < 100s for 8 steps (< 12.5s/step). Stretch: < 40s (< 5s/step).

5. **Memory check**: peak RSS during MLX denoise must stay under 48 GB unified memory budget.

## Integration with existing infrastructure

- **Tracer**: MlxBackend emits the same stage names as MpsBackend. X-ray, Perfetto export, diff-traces all work unchanged.
- **CLI**: `decomposer decompose --backend mlx ...`
- **Web**: `DECOMPOSER_BACKEND=mlx` in .env
- **Auto-researcher**: can test MlxBackend experiments via a `backend_switch` hypothesis kind
- **Doctor**: `decomposer doctor --backend mlx` validates the MLX path

## Dependencies

```toml
"mlx>=0.22.0",
"mlx-nn>=0.22.0",
```

MLX is Apple-Silicon-only. On non-Apple platforms, `import mlx` will fail. MlxBackend should be importable only when MLX is available — use lazy import pattern (same as MpsBackend's lazy diffusers import).

## Build order

1. Scaffold: `mlx_dit/` package, `mlx_backend.py` skeleton, Settings fields, CLI `--backend` flag
2. Converter: GGUF → MLX quantized safetensors
3. Layers: RMSNorm, MLP, modulation in MLX
4. Attention: MultiHeadAttention + 3D RoPE
5. MMDiTBlock: attention + FFN + AdaLayerNorm
6. Transformer: stack 60 blocks + embeddings + final layer
7. Loader: load MLX safetensors → populate transformer
8. MlxBackend: wire MLX denoise loop + MpsBackend delegation
9. Integration test: end-to-end decomposition
10. Performance tuning: `mx.compile`, memory optimization

## Open risks

| Risk | Mitigation |
|---|---|
| 3D RoPE implementation differs subtly from diffusers | Side-by-side numerical comparison test with identical inputs |
| VLD (variable-layer decomposition) conditioning is more complex than documented | Read diffusers source for `prepare_latents` layered logic; may need to reimplement |
| `mx.fast.scaled_dot_product_attention` doesn't support the attention mask pattern Qwen-Image uses | Fall back to manual Q@K.T → softmax → @V; still faster than PyTorch MPS because matmuls are quantized |
| MLX quantization format differs from GGUF quality at same bit width | Converter validates round-trip error; accept MLX-native quantization even if slightly different from GGUF |
| Peak memory during MLX inference exceeds 48 GB | 4-bit quantization keeps weights at ~10 GB; MLX lazy evaluation should bound activations |
| mflux is FLUX-specific; Qwen-Image MMDiT has structural differences | mflux is a template for the pattern, not a copy-paste; expect 60% reusable, 40% custom |

## Success criteria

- `decomposer decompose test_image.jpg --backend mlx --layers 3 --steps 8` produces valid RGBA layers
- `denoise_loop.wall_ms` < 100s total (< 12.5s/step) — 6.5× improvement over current 829s
- Stretch: < 40s total (< 5s/step) — 20× improvement
- Auto-researcher oracle: structural similarity ≥ thresholds vs MpsBackend baseline
- No PyTorch dependency in the denoise hot path
- Peak memory < 48 GB

## Estimated timeline

- **Week 1**: Tasks 1-4 (scaffold, converter, layers, attention)
- **Week 2**: Tasks 5-7 (blocks, transformer, loader)
- **Week 3**: Tasks 8-9 (backend wiring, integration test)
- **Week 4**: Task 10 (performance tuning, stretch goals)
