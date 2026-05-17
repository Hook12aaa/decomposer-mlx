# decomposer

A Python inference pipeline for [Qwen-Image-Layered](https://huggingface.co/Qwen/Qwen-Image-Layered) on Apple Silicon. Image in, RGBA layers out, no cloud dependency. Runs on M3 Max in ~8 minutes with 21 GB peak memory using MLX 8-bit quantised matmul.

The interesting part is not the pipeline. It is the four bugs I found when porting from PyTorch to MLX, each invisible without the last, and the staged checkpoint comparison technique that found them. [RESEARCH.md](RESEARCH.md) documents the full forensic process.

**Status:** 0.2.0 · MIT · Apple Silicon only · Python 3.12+ · macOS 14+

## Why this exists

Marketing teams extract individual components from flat banner images (the hero photo, the logo, the CTA button, the background) to remix them across formats. Doing this manually in Photoshop takes 15-30 minutes per asset. Qwen-Image-Layered is a 20B-parameter diffusion model trained specifically for this decomposition task. I wanted to run it locally on an M3 Max with no cloud dependency.

After building [qwen3-mlx](../inference/) for text inference and [marlin-mlx](../video/) for video captioning, I wanted to know if the same bandwidth-first optimisation methodology would apply to a diffusion model. The answer was yes, but the debugging journey was harder. The diffusion model's output was a visual image, not tokens, so "wrong output" looked like noise rather than garbled text. I had to build a staged checkpoint comparison tool to trace where the PyTorch and MLX pipelines diverged, token by token.

## Performance

M3 Max, Qwen-Image-Layered, 3 layers, 640px, 8 steps:

| Engine | Total | Denoise/step | Peak memory | vs MPS |
|---|---|---|---|---|
| PyTorch MPS fp32 (baseline) | 1048s (17 min) | 92s | 46 GB | 1.00× |
| PyTorch MPS + auto-researcher wins | ~862s (14 min) | ~82s | ~40 GB | 1.22× |
| **MLX 8-bit** | **458s (7.6 min)** | **~40s** | **21 GB** | **2.29×** |

37 tests pass. Output layers match MPS quality: background separated from hero image separated from text/logo overlay. Quality validated by SSIM diversity + alpha coverage checks.

## What Qwen-Image-Layered actually is

The HuggingFace card says it is a 20B-parameter diffusion transformer for image layer decomposition. Here is what it does not say.

### The architecture

Three subsystems, not one:

```
Image (JPEG) --> Qwen2.5-VL-7B --> prompt_embeds (2291 tokens × 3584)
                                          |
Image (JPEG) --> VAE encode --> latent (1 × 16 × 1 × 92 × 72)
                                          |
                     60-block MMDiT (20B params)
                     24 heads × 128 head_dim
                     joint attention (image + text)
                     3D RoPE (frame × height × width)
                     flow-match Euler denoising (8 steps)
                                          |
                     VAE decode --> N RGBA layer PNGs
```

The text encoder is Qwen2.5-VL-7B, a full 7B-parameter vision-language model, not a CLIP text encoder. It runs multimodal: it sees the input image through its vision tower and produces 2291 conditioning tokens.

The DiT is an MMDiT (multi-modal diffusion transformer) with dual-stream modulation. Each block processes image tokens and text tokens jointly through attention, with separate modulation parameters for each stream.

### Where the time goes

I measured this from the X-ray tracer. Every decomposition reads these stages:

| Stage | Time | % of total |
|---|---|---|
| Text encoder (Qwen2.5-VL-7B, fp32 MPS) | 328s | 56% |
| Denoise loop (60-block MMDiT, 8 steps) | 330s/130s | 34% |
| VAE decode (CPU conv3d fallback) | 30s | 8% |
| Model load + latent packing | 7s | 2% |

The text encoder dominates. It is a 7B-parameter VLM running in fp32 on PyTorch MPS. The 20B DiT, the part I ported to MLX, is only 34% of total time.

### Things I learned the hard way

These are the details that cost debugging time because no documentation mentions them.

**1. fp16 overflows in attention.** The MMDiT's attention logits routinely exceed fp16's ±65504 range. Every fp16 run produces NaN. Every pixel comes out transparent. The fix is fp32, but the symptom (all-transparent output) is misleading. I initially suspected the GGUF dequant, then the VAE, then the pipeline callback. The actual cause was in the attention dot product.

**2. The GGUF dequant is the bandwidth bottleneck.** `GgufLinear.forward()` dequantizes Q8 buffers to fp32 in Python (`quants * scales`), runs `F.linear`, then discards the fp32 tensor. The GPU never sees quantized data. On M3 Max at ~300 GB/s, the 21 GB DiT should transfer in ~70 ms/step. Observed: 103,700 ms/step. 1000× off the bandwidth ceiling. MLX's fused quantized matmul eliminates the Python-level dequant entirely.

**3. The RoPE is not standard.** Qwen-Image-Layered uses `Layer3DRope` with three features not present in FLUX's RoPE: per-layer frame identity (each noise frame gets its layer index as temporal position), centered spatial frequencies via negative indices (`cat(neg[-tail:], pos[:head])`), and a separate negative-frequency encoding for the conditioning image. Using FLUX's flat 3D RoPE produces noise that looks structured but contains no decomposition signal.

**4. The sigma schedule endpoint matters.** The flow-match scheduler generates base sigmas from `linspace(1.0, 1/num_train_timesteps)` (endpoint 0.001). Using `linspace(1.0, 1/num_steps)` (endpoint 0.125) leaves 20% noise in the final output because the schedule never reaches near-zero.

**5. The timestep scale is 1000×.** The sinusoidal timestep embedding multiplies the input by `scale=1000` before computing frequencies. Without this, the model receives the wrong temporal conditioning at every step. The cosine similarity between correct and incorrect timestep embeddings was 0.265, essentially uncorrelated.

**6. 4-bit quantisation is too lossy for this model.** The MLX 4-bit DiT produces noise even with correct scheduling and RoPE. The same model at 8-bit produces clean decomposition matching PyTorch Q8 quality. This is different from text generation models (Qwen3-4B runs fine at 4-bit) and likely relates to the precision required for spatial alpha-channel prediction.

**7. conv3d has no MPS kernel.** The VAE uses 3D convolutions. PyTorch's MPS backend does not implement `aten::slow_conv3d_forward`. Setting `PYTORCH_ENABLE_MPS_FALLBACK=1` routes to CPU, but then downstream ops expect MPS tensors. The fix is to run the entire VAE decode phase on CPU.

## Setup

```bash
uv sync --extra dev
uv run decomposer doctor --fake    # validate install (no model load)
uv run decomposer doctor           # validate MPS path (downloads ~50 GB first time)
```

Requires Python 3.12+, macOS 14+ on Apple Silicon, ~50 GB free disk for model weights.

### MLX weights (one-time conversion)

```bash
uv run decomposer convert-to-mlx <gguf-path> --output mlx-weights-8bit --bits 8
```

This produces `mlx-weights-8bit/` (~21 GB, 10 shards). Set `DECOMPOSER_MLX_WEIGHTS_DIR=mlx-weights-8bit` or update your `.env`. The 8-bit conversion takes ~90 seconds.

## Run

**Primary command:**

```bash
decomposer run photo.jpg                          # 6 layers, MLX, smart defaults
decomposer run photo.jpg --layers 3               # fewer layers
decomposer run photo.jpg -o ./my-output           # custom output dir
decomposer run banner.png hero.jpg ad.webp        # batch: multiple images
decomposer run photo.jpg --trace                  # include performance trace
decomposer run photo.jpg --seed 42                # reproducible output
decomposer run photo.jpg --backend mps            # use PyTorch MPS instead
decomposer run photo.jpg --backend fake           # mock (instant, for testing)
```

Each run produces:

```
output/<image_stem>/
├── layer_0.png          # RGBA PNG
├── layer_1.png
├── ...
├── manifest.json        # classification, metrics, quality warnings
└── trace.json           # performance trace (with --trace)
```

**Other commands:**

```bash
decomposer doctor --fake              # validate install without model load
decomposer convert-to-mlx <gguf>     # one-time GGUF → MLX conversion
decomposer diff-traces a.json b.json  # compare two performance traces
decomposer research baseline          # run auto-researcher baseline
```

**Web UI:**

```bash
uv run uvicorn decomposer.web.app:app --host 127.0.0.1 --port 8000
```

## How it works

```
    decomposer run image.jpg
            │
            ▼
┌─────────────────────────┐
│  Preprocess              │
│  validate, convert RGBA  │
│  reject < 64px           │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Preflight               │
│  check MLX weights exist │
│  warn if memory < 20 GB  │
└───────────┬─────────────┘
            ▼
┌─────────────────────────────────────────┐
│  Inference Pipeline                      │
│  ┌────────────────────────────────────┐ │
│  │ Text encoder (PyTorch MPS, fp32)  │ │  ← 56% of time
│  │ Qwen2.5-VL-7B, 2291 tokens       │ │
│  ├────────────────────────────────────┤ │
│  │ VAE encode (PyTorch MPS)          │ │
│  │ → latent (1×16×1×92×72)           │ │
│  ├────────────────────────────────────┤ │
│  │ Denoise loop (MLX, 8-bit fused)   │ │  ← 34% of time
│  │ 60-block MMDiT, mx.compile        │ │
│  │ Layer3D RoPE, joint attention     │ │
│  │ FlowMatch Euler, 8 steps          │ │
│  ├────────────────────────────────────┤ │
│  │ VAE decode (PyTorch CPU)          │ │  ← 8% of time
│  │ conv3d → CPU fallback             │ │
│  └────────────────────────────────────┘ │
└───────────┬─────────────────────────────┘
            ▼
┌─────────────────────────┐
│  Quality Validation      │
│  non-degeneracy, SSIM    │
│  alpha coverage          │
└───────────┬─────────────┘
            ▼
┌─────────────────────────┐
│  Classify + Export       │
│  heuristic labels        │
│  PNGs + manifest.json    │
└─────────────────────────┘
```

**manifest.json:** every layer is classified heuristically (background, hero_image, logo, text, cta_button, overlay, unknown) with confidence scores, alpha coverage, bounding box, and mean RGB. Quality warnings flag degenerate output.

## Repository layout

```
.
├── README.md
├── RESEARCH.md                        optimisation journey + forensic log
├── pyproject.toml
├── Dockerfile
├── decomposer/
│   ├── cli.py                         typer CLI: run, decompose, doctor, convert-to-mlx
│   ├── config.py                      pydantic-settings (DECOMPOSER_* env vars)
│   ├── preprocess.py                  input validation, RGBA conversion, size checks
│   ├── classifier.py                  heuristic layer classification (7 categories)
│   ├── manifest.py                    pydantic manifest schema + JSON writer
│   ├── quality.py                     post-decomposition quality checks (SSIM, alpha)
│   ├── workflow.py                    orchestrator: preprocess → inference → classify → export
│   ├── debug_compare.py               staged MLX vs MPS checkpoint comparison
│   ├── logging_setup.py
│   ├── core/
│   │   ├── backend.py                 InferenceBackend protocol, FakeBackend, get_backend()
│   │   ├── mps_backend.py             PyTorch MPS: text encoder + VAE + DiT (fp32, Q8)
│   │   ├── mlx_backend.py             MLX: DiT denoise, delegates encoder/VAE to MPS
│   │   ├── gguf_loader.py             GgufLinear + Q8/Q4/Q5 dequant
│   │   ├── gguf_pipeline.py           GGUF → diffusers transformer loader
│   │   ├── residency.py               at-most-one MPS module enforcer
│   │   ├── xray.py                    per-run tracer with nested stages
│   │   ├── probes.py                  MPS/RSS memory probes
│   │   ├── perfetto.py                trace → Perfetto JSON export
│   │   └── types.py                   StageRecord, Report dataclasses
│   ├── mlx_dit/                       pure MLX transformer
│   │   ├── transformer.py             60-block MMDiT (24 heads, 128 head_dim)
│   │   ├── transformer_block.py       dual-stream modulation block
│   │   ├── attention.py               joint attention + mx.fast.sdpa
│   │   ├── rope.py                    Layer3D RoPE (per-layer identity, centered spatial)
│   │   ├── layers.py                  RMSNorm, FeedForward, TimestepEmbedding (scale=1000)
│   │   ├── loader.py                  load quantized safetensors shards (auto-detects bits)
│   │   └── scheduler.py               FlowMatch Euler discrete (Qwen-Image mu schedule)
│   ├── mlx_convert/
│   │   └── convert.py                 GGUF → MLX 4/8-bit (sharded saves, bits in config)
│   ├── research/                      auto-researcher framework
│   │   ├── oracle.py                  SSIM + Hungarian matching + non-degeneracy
│   │   ├── runner.py                  experiment execution + worktree isolation
│   │   ├── run.py                     experiment queue runner
│   │   ├── experiments.py             hypothesis definitions
│   │   ├── ledger.py                  experiment results ledger
│   │   ├── apply.py                   hypothesis applicator
│   │   ├── patches/                   individual optimisation patches
│   │   └── cli.py                     research subcommands
│   └── web/
│       ├── app.py                     FastAPI: POST /jobs, SSE, ZIP, /healthz
│       ├── jobs.py                    JobStore + SqliteJobStore
│       └── templates/index.html
├── tests/                             37 tests, all passing
├── mlx-weights-8bit/                  MLX 8-bit safetensors (21 GB, 10 shards)
└── docs/superpowers/
    ├── specs/                         design documents
    └── plans/                         implementation plans
```

## Key principles

1. **Bandwidth before code.** The 1000x gap between theoretical and observed throughput was not in the code. It was in how data reached the GPU. MLX's fused quantised matmul eliminated the Python-level dequant bottleneck.
2. **Measure before optimising.** Three of six auto-researcher experiments failed. Each was caught by the quality oracle before merging. The three that succeeded were verified by SSIM comparison against a known-good baseline.
3. **Stage gates for debugging.** When the MLX output was wrong, dumping latent tensor statistics at every pipeline boundary (text encoder output, VAE latent, per-step denoise output) pinpointed the exact divergence point. Four bugs found in one session.
4. **8-bit minimum for diffusion decomposition.** Unlike text generation models that tolerate 4-bit quantisation, this decomposition model requires 8-bit precision. The alpha-channel predictions are too sensitive to quantisation noise at 4-bit.
5. **Dead ends are documentation.** RESEARCH.md records every failed optimisation attempt with the same rigour as the successes. The next person who considers Lightning LoRA, Q5_K_M dequant, or 4-bit MLX for this model will find the evidence that it does not work and why.

## Known limitations

The text encoder (56% of wall time) still runs on PyTorch MPS in fp32. Porting it to MLX would require implementing a full Qwen2.5-VL-7B pipeline including the vision tower, which is significantly more complex than the DiT port.

The VAE runs on CPU because MPS lacks conv3d. Decomposing conv3d into per-frame conv2d operations would allow MPS acceleration, but the VAE is only 8% of total time.

The auto-researcher experiments were validated against a single test image. A diverse corpus of marketing assets (different layouts, text densities, logo placements) would make the quality gate more reliable.

## HuggingFace authentication

The text encoder and GGUF transformer live on gated HuggingFace repos. Accept each repo's license on huggingface.co and authenticate locally:

```bash
huggingface-cli login                       # writes token to ~/.cache/huggingface/token
# OR
export DECOMPOSER_HF_TOKEN=hf_xxxxx        # respected by all decomposer entry points
```

`decomposer doctor` probes each required repo at startup and prints a remediation checklist if access is missing.

## Troubleshooting

- **All output layers are transparent** → compute dtype is fp16. The model overflows fp16 in attention. This should not happen with current defaults (fp32), but if you have changed settings, switch back.
- **MLX output is noisy/grid pattern** → you are running 4-bit weights. This model requires 8-bit minimum. Convert with `--bits 8`.
- **MLX weights not found** → run `decomposer convert-to-mlx <gguf_path> --output mlx-weights-8bit --bits 8`.
- **OOM during inference** → reduce resolution (`--resolution 512`) or layers (`--layers 3`). Peak memory is ~21 GB for MLX 8-bit, ~46 GB for MPS.
- **HF auth failure** → run `huggingface-cli login` or set `DECOMPOSER_HF_TOKEN`. Confirm you have accepted the license on each gated repo URL.
- **MPS not available** → confirm Apple Silicon hardware, macOS 14+, torch with MPS support.
- **Inference timeout** → the web app cancels jobs exceeding `DECOMPOSER_INFERENCE_TIMEOUT_SECONDS` (default 600s). Increase the budget or reduce parameters.

## Licence

MIT.
