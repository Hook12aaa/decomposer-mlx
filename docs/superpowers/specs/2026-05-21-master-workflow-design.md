# Master Workflow: MLX-First Image Decomposition Pipeline

## Goal

A single `decomposer run image.jpg` command that reliably decomposes any marketing image into classified RGBA layers with metadata, running locally on M3 Max (48GB) in ~5 minutes using the MLX 4-bit backend.

## Architecture

Five stages, chained sequentially:

1. **Input preprocessor** — validate image, convert to RGBA, check dimensions
2. **Pre-flight checks** — verify MLX weights, HF cache, available memory
3. **MLX inference** — text encode → image encode → 4-bit denoise → VAE decode
4. **Layer classifier** — heuristic post-processing on output layers
5. **Output writer** — PNGs + manifest.json + optional trace

## CLI Interface

### Primary command: `decomposer run`

```
decomposer run photo.jpg                    # Smart defaults: 6 layers, 640px, 8 steps
decomposer run photo.jpg --layers 4         # Override layer count
decomposer run photo.jpg -o ./my-output     # Custom output dir
decomposer run banner.png hero.jpg ad.webp  # Batch: multiple images
decomposer run photo.jpg --trace            # Include performance trace
decomposer run photo.jpg --seed 42          # Reproducible output
```

**Defaults:** 6 layers, 640px resolution, 8 steps, MLX backend, output to `./output/<image_stem>/`.

**Supported input formats:** JPEG, PNG, WebP, TIFF, BMP — any format PIL can load.

**Validation:**
- Reject images smaller than 64x64
- Warn if image exceeds 2048x2048 (will be downscaled)
- Fail fast if image is corrupt/unloadable

The existing `decomposer decompose` command remains unchanged for backwards compatibility.

## Input Preprocessing

Module: `decomposer/preprocess.py`

Responsibilities:
- Load image via PIL, fail with clear message if corrupt
- Convert to RGBA (add opaque alpha channel if RGB/grayscale)
- Validate minimum dimensions (64x64)
- Log warning for oversized images (>2048px on either axis)
- Return validated PIL.Image.Image in RGBA mode

## Pre-flight Checks

Integrated into `decomposer/workflow.py` before inference starts:

1. **MLX weights:** Verify `mlx_weights_dir` exists and contains safetensors shards
2. **HF cache:** Check text encoder and VAE are cached (or trigger download)
3. **Memory:** Warn if system available memory < 20GB (pipeline peaks at ~16GB)

Each check fails fast with an actionable error message. No 6-minute pipeline runs that were doomed from the start.

## MLX Inference Pipeline

Uses the existing `MlxBackend.decompose()` with these enhancements:

### Proven optimizations to integrate

| Optimization | Effect | Source |
|---|---|---|
| MLX 4-bit fused matmul | 3.2x denoise speedup | Already in MlxBackend |
| `mx.compile` on denoise loop | Expected 10-30% further speedup | New — apply to MLXQwenTransformer.__call__ |
| bf16 autocast for text encoder + VAE | ~20% encode/decode speedup | Proven in auto-researcher, port to MLX path |
| Empty cache between stages | ~16% memory reduction | Proven in auto-researcher, port to MLX path |

### Performance targets

| Stage | Current (MLX) | Target |
|---|---|---|
| Text encoder load + encode | 213s | ~180s |
| Denoise (8 steps) | 130s (32.5s/step) | ~100s (~12.5s/step with mx.compile) |
| VAE decode | 30s | ~25s |
| Total | 380s (6.3 min) | ~300s (5 min) |

## Layer Classification

Module: `decomposer/classifier.py`

Heuristic-based classification applied to output RGBA layers. Runs in milliseconds.

### Classification categories

| Category | Heuristic |
|---|---|
| `background` | Alpha coverage > 95%, largest bounding box |
| `hero_image` | Alpha coverage 20-80%, largest non-background area |
| `text` | High contrast edges, narrow bounding box height, low alpha coverage |
| `logo` | Small bounding box area (<15% of image), compact shape |
| `cta_button` | Rectangular, small area, high saturation, bottom-half position |
| `overlay` | Semi-transparent (alpha 5-30%), spans large area |
| `unknown` | No strong match |

Each classification includes a confidence score (0.0-1.0) based on how strongly metrics match thresholds.

### Per-layer metrics computed

- `alpha_coverage`: fraction of non-transparent pixels
- `bounding_box`: [x_min, y_min, x_max, y_max] of non-transparent region
- `mean_rgb`: average color of non-transparent pixels
- `file_size_bytes`: output PNG file size

## Output Format

Output directory structure per image:

```
output/<image_stem>/
├── layer_0.png
├── layer_1.png
├── ...
├── manifest.json
└── trace.json          # Only with --trace flag
```

### manifest.json schema

```json
{
  "source": "banner.jpg",
  "source_dimensions": [1920, 1080],
  "resolution": 640,
  "steps": 8,
  "layers_requested": 6,
  "backend": "mlx",
  "seed": null,
  "wall_time_seconds": 300.5,
  "quality_warnings": [],
  "layers": [
    {
      "file": "layer_0.png",
      "index": 0,
      "classification": "background",
      "confidence": 0.92,
      "alpha_coverage": 1.0,
      "bounding_box": [0, 0, 640, 480],
      "mean_rgb": [45, 82, 130],
      "file_size_bytes": 1148576
    }
  ]
}
```

## Quality Validation

Post-decomposition checks before writing output:

1. **Non-degeneracy:** At least one layer must have alpha coverage < 90% (all-opaque = failed decomposition)
2. **Diversity:** Pairwise SSIM between layers must be < 0.95 (identical layers = degenerate)
3. **Alpha sanity:** No layer should be all-transparent

Failed checks are logged as warnings in `manifest.json["quality_warnings"]` — output is still written but flagged.

## Error Handling

| Scenario | Response |
|---|---|
| MLX weights missing | Fail with: "Run `decomposer convert-to-mlx <gguf_path>` first" |
| HF model not cached | Trigger download, show progress |
| OOM during denoise | Catch, report peak memory, suggest reducing resolution |
| NaN in output | Detect before VAE decode, report diverged step |
| Ctrl+C interrupt | Clean partial output directory |
| Corrupt input image | Fail fast before pipeline starts |
| Memory < 20GB available | Warn but proceed (may OOM) |

## MLX Quality Investigation

Known gap: MLX layers show 99-100% opaque vs PyTorch's varied 100%/96%/81% distribution.

Investigation plan:
1. Run MLX and MPS on same image with same seed
2. Compare latent distributions before VAE decode
3. If issue is in scheduler sigma handling or noise init: fix in MLX
4. If inherent to 4-bit quantization: document as known tradeoff

## File Map

| File | Responsibility |
|---|---|
| `decomposer/workflow.py` | New — orchestrates preprocess → preflight → inference → classify → export |
| `decomposer/preprocess.py` | New — input validation and image preparation |
| `decomposer/classifier.py` | New — heuristic layer classification |
| `decomposer/manifest.py` | New — manifest.json schema (pydantic) and writer |
| `decomposer/cli.py` | Modify — add `run` command |
| `decomposer/core/mlx_backend.py` | Modify — integrate mx.compile, bf16 for PyTorch stages |
| `decomposer/config.py` | Modify — add any new settings if needed |

## Non-goals

- Cloud deployment (local M3 Max only)
- Web UI changes (CLI-primary)
- PSD export (PNGs + JSON only)
- VLM-based classification (heuristic only)
- New model training or fine-tuning
