# Research Log

How a Qwen-Image-Layered decomposition pipeline went from broken (all-transparent NaN output) to 458s on an M3 Max, and the four bugs in the MLX port that were each invisible without the last.

## The starting point

Qwen-Image-Layered is a 20B-parameter MMDiT diffusion model that decomposes a flat marketing image into RGBA layers (hero image, background, logo, text, CTA). HuggingFace hosts a GGUF quantised checkpoint. The goal was to run it locally on an M3 Max (48 GB unified memory) as a CLI tool.

The first working pipeline loaded the Q8 GGUF transformer via a custom `GgufLinear` module (dequantise per-forward to avoid 40 GB fp16 inflation), a bf16 text encoder (Qwen2.5-VL-7B), and the Qwen-Image VAE. All stages ran on MPS through diffusers' `QwenImageLayeredPipeline`.

Every output was all-transparent. Every pixel had NaN alpha.

## The NaN problem

The default compute dtype was fp16. The MMDiT attention layers produce activation values that overflow fp16's ±65504 range. Switching to fp32 fixed everything. Real RGBA layers appeared immediately.

This was not subtle. The model is a 20B-param transformer; its intermediate activations routinely exceed fp16 range in the attention logits. Every fp16 run produces NaN. The fix was one line: change the default dtype from `torch.float16` to `torch.float32`.

The cost was significant. fp32 is roughly 2× slower than fp16 on MPS for matmul. The first working run took 17 minutes (1034s) for 8 denoise steps at 640px, 3 layers. 103.7s per step. 46 GB peak memory.

## What the auto-researcher found

An auto-researcher framework ran experiments in git worktrees, measured quality via SSIM + Hungarian matching + non-degeneracy checks, and auto-merged winners. Three experiments landed:

| Experiment | Wall reduction | How |
|---|---|---|
| FBCache (skip redundant transformer blocks) | -14.3% denoise | diffusers' `FirstBlockCacheConfig(threshold=0.08)` skips blocks whose output hasn't changed since the previous step |
| Empty MPS cache between stages | -16.6% peak memory | `torch.mps.empty_cache()` after freeing each model stage releases MPS allocator fragmentation |
| bf16 autocast for denoise | -11.7% denoise | `torch.autocast(device_type='mps', dtype=torch.bfloat16)`. bf16 has fp32's exponent range (no NaN) with fp16's speed |

Three experiments failed:

| Experiment | Result | Why |
|---|---|---|
| Lightning LoRA (step reduction) | Blocked | PEFT cannot adapt `GgufLinear`, only works on `nn.Linear`. Would need LoRA-merged GGUF checkpoint |
| Q5_K_M quantisation | Degenerate output | K-quant dequant implementation produces correct output on synthetic test data but corrupt output on real GGUF data. Unresolved bug in the d_min/d_scale path |
| Q4_K_M quantisation | Degenerate output | Same k-quant dequant issue as Q5 |

After auto-researcher: ~862s (14 min), ~40 GB peak.

## The MLX port

The critical observation came from bandwidth analysis. M3 Max sustains ~300 GB/s memory bandwidth. The 21 GB Q8 DiT should transfer in ~70 ms per step. Measured: 103,700 ms per step. 1000× off the bandwidth ceiling.

The bottleneck was `GgufLinear.forward()`: every call dequantises Q8 buffers to fp16/fp32 tensors in Python (`quants * scales`), then runs `F.linear`. The dequantised weight is allocated, computed, used once, and discarded. The Metal GPU never sees quantised data. It only sees the fully materialised fp32 tensor.

MLX solves this with fused quantised matmul. `mlx.core.quantized_matmul` reads packed quantised bytes directly in the Metal kernel and dequantises per-element during the matmul. No Python-level dequant. No full-precision weight allocation.

The port adapted mflux (an existing MLX port of FLUX, same MMDiT architecture family). The MlxBackend delegates text encoder and VAE to PyTorch MPS (not worth porting, under 15% of wall time) and runs only the denoise loop in MLX.

## The four bugs

The MLX port ran end-to-end on the first attempt. The output was noise. Not a crash, not NaN. Structured noise that looked like a failed decomposition. Finding the cause took longer than building the port.

### Bug 1: Wrong mu schedule (FLUX formula on a Qwen-Image model)

The mu parameter controls the dynamic sigma shift for the flow-match scheduler. mflux (FLUX) uses `mu = sqrt(seq_len / base_seq_len)`. Qwen-Image uses a linear interpolation: `mu = m * seq_len + b` where `m = (max_shift - base_shift) / (max_seq_len - base_seq_len)`.

For 1656 image patches: FLUX mu = 2.54, Qwen-Image mu = 0.57. With the wrong mu, the sigma schedule barely decreases (1.0 → 0.65 instead of 1.0 → 0.20). The model sees mostly-noisy inputs at every step and never properly denoises.

I found this by computing what diffusers' `FlowMatchEulerDiscreteScheduler` actually produces and comparing the sigma arrays side by side. The mu computation was a copy from the mflux template that I never questioned.

### Bug 2: Wrong sigma endpoint (1/num_steps instead of 1/num_train_timesteps)

The base sigma schedule used `np.linspace(1.0, 1/num_steps, num_steps)` (endpoint 0.125 for 8 steps). Diffusers uses `np.linspace(1.0, 1/num_train_timesteps, num_steps)` (endpoint 0.001). After the time shift, the final sigma was 0.20 (20% noise remaining) instead of 0.002 (fully denoised).

I found this by running the staged checkpoint comparison: the per-step latent std in MLX decreased too fast (0.95 → 0.34 by step 4) then rebounded (back up to 0.45 by step 7). The MPS trajectory decreased smoothly (0.95 → 0.37). The rebound was the signature of an incomplete schedule. The model started adding noise back because the sigma never reached near-zero.

### Bug 3: Missing timestep scale (×1000)

The sinusoidal timestep embedding multiplies the input by `scale=1000` before computing frequencies. The PyTorch `Timesteps` module has `self.scale = 1000` and applies it in forward. My MLX `MLXTimesteps` did not have this parameter.

Without the scale, timestep 0.5 produces frequencies for position 0.5. With the scale, it produces frequencies for position 500. The cosine similarity between the two outputs was 0.265, essentially uncorrelated random vectors. Every block in the transformer received the wrong temporal conditioning.

I found this by running a gate-by-gate functional comparison: feeding the same test tensor through both models' `time_proj` module and checking cosine similarity. Gate 1 (img_in) passed at 0.997. Gate 2 (time_proj) failed at 0.265. The divergence was at the very first step of the timestep embedding.

### Bug 4: Wrong RoPE (FLUX flat grid vs Qwen-Image Layer3DRope)

The FLUX-style RoPE I copied from mflux uses a flat 3D grid: all frames share sequential temporal indices, spatial frequencies are simple `pos_freqs[:height]`, and there is no distinction between noise frames and the conditioning image.

Qwen-Image's `Layer3DRope` has three differences:

1. **Per-layer frame identity.** Each noise frame gets its layer index (0, 1, 2, 3) as the temporal position. The model uses this to distinguish which frame corresponds to which output layer.

2. **Centered spatial frequencies.** When `scale_rope=True`, height and width use `cat(neg_freqs[-tail:], pos_freqs[:head])`, frequencies centered around zero via negative indices. This is not cosmetic; the training data used this representation.

3. **Negative-frequency conditioning.** The conditioning image (the input photo being decomposed) uses `neg_freqs[-1]` for its frame position, placing it in negative frequency space, separate from all noise frames.

I found this by reading the diffusers `QwenEmbedLayer3DRope` source after all other components (img_in, timestep embedding, modulation, attention projections) passed the gate check at cos > 0.999. The RoPE was the last untested component. Comparing the full RoPE output (8280 tokens × 64 frequencies) between PyTorch and my MLX implementation showed the divergence.

After fixing: cosine similarity 0.99999+ on all five frames.

### Why four bugs, not one

Each bug was invisible without the previous fixes:

- Bug 1 (wrong mu) made the output look like random noise. Fixed: output changed to structured grid noise.
- Bug 2 (wrong sigma endpoint) left residual noise. Fixed: grid pattern changed but still not decomposition.
- Bug 3 (missing timestep scale) corrupted every block's conditioning. Fixed: all gate checks passed, output changed again.
- Bug 4 (wrong RoPE) made the model unable to distinguish layers. Fixed: decomposition still produced noise at 4-bit.

The final discovery: 4-bit quantisation is too lossy for this model. Switching to 8-bit with all four fixes produced clean decomposition matching PyTorch quality.

I could not have found bugs 2-4 without fixing bug 1 first, because the wrong mu made the entire sigma schedule meaningless. And I could not have found the 4-bit precision issue without fixing all four bugs, because any one of them alone produced noise indistinguishable from quantisation artifacts.

## The staged checkpoint comparison

The technique that found bugs 2-4 was a side-by-side pipeline comparison with per-stage artifact dumps.

Both backends (MPS as ground truth, MLX as experiment) were run on the same image with the same seed. At each boundary point (text encoder output, VAE latent, per-step denoise latents, final latents), both backends' tensors were saved as numpy arrays and compared.

```
Checkpoint          MPS mean    MLX mean    MPS std     MLX std     Verdict
─────────────────────────────────────────────────────────────────────────
cond                -0.1247     -0.1247     3.9303      3.9303      MATCH
image_latent         0.0583      0.0583     0.3778      0.3778      MATCH
step_0               0.0037      0.0062     0.9460      0.7921      OK
step_1               0.0073      0.0098     0.8832      0.6346      OK
step_2               0.0116      0.0116     0.8090      0.5053      OK
step_3               0.0169      0.0147     0.7210      0.3989      OK
step_4               0.0235      0.0163     0.6162      0.3423      OK
step_5               0.0320      0.0260     0.4949      0.3554      OK
step_6               0.0433      0.0375     0.3761      0.3933      OK
step_7               0.0589      0.0494     0.3740      0.4492      DIVERGED
```

The shared stages (cond, image_latent) were identical, confirming the text encoder and VAE paths are correct. The divergence accumulated through the denoise steps: MLX denoised too fast (std dropped to 0.34 by step 4) then rebounded (back to 0.45 by step 7), while MPS converged smoothly (0.95 → 0.37). This pattern pointed directly at the sigma schedule.

The gate-by-gate functional comparison then isolated each component: feeding the same test tensor through both models' `img_in`, `time_proj`, `timestep_embedder`, and `block_0` independently. Each gate produced a cosine similarity verdict. The first gate that failed (time_proj at cos=0.265) identified the timestep scale bug.

## The numbers

M3 Max, Qwen-Image-Layered, 3 layers, 640px, 8 steps:

| Stage | fp16 (broken) | fp32 MPS | MPS + optimised | MLX 4-bit | MLX 8-bit |
|---|---|---|---|---|---|
| Denoise/step | NaN | 103.7s | ~82s | 32.5s | ~40s |
| Total wall | N/A | 1048s | ~862s | noise | 458s |
| Peak memory | — | 46 GB | ~40 GB | 16 GB | 21 GB |
| Output quality | Transparent | Clean | Clean | Noise | Clean |

MLX 8-bit: 2.29× faster than MPS, clean decomposition, 54% less memory.

## What worked (summary)

| Fix | Effect | How found |
|---|---|---|
| fp32 dtype | All-transparent → real layers | Identified NaN in attention logits |
| FBCache | -14.3% denoise | Auto-researcher quality gate |
| Empty MPS cache | -16.6% peak memory | Auto-researcher quality gate |
| bf16 autocast | -11.7% denoise | Auto-researcher quality gate |
| MLX fused quantised matmul | 3.2× denoise speedup | Bandwidth analysis (1000× off ceiling) |
| Correct mu schedule | Structured noise → different noise | Compared diffusers sigma array |
| Correct sigma endpoint | Different noise → still wrong | Staged checkpoint comparison |
| Timestep ×1000 scale | Gate check cos 0.265 → 1.0 | Gate-by-gate functional comparison |
| Layer3D RoPE rewrite | Gate check cos 0.999+ | Diffusers source reading |
| 8-bit instead of 4-bit | Noise → clean decomposition | Elimination after all bugs fixed |

## What did not work

| Attempt | Result | Why |
|---|---|---|
| Lightning LoRA | Blocked | PEFT cannot adapt GgufLinear |
| Q5_K_M quant | Degenerate | K-quant dequant bug on real GGUF data |
| Q4_K_M quant | Degenerate | Same k-quant bug |
| MLX 4-bit | Noise | Quantisation too lossy for decomposition |
| Compiled recurrence (for SSM, other project) | Neutral | Matmuls dominate, elementwise ops are not the bottleneck |

## What I learned that the text and video projects did not teach me

The inference project (Qwen3-4B) taught me that source-level forensics produce wrong hypotheses and that bandwidth analysis finds the real bottleneck. The video project (Marlin-2B) taught me that SSM state differs from KV cache, mixed-precision quantisation is mandatory for hybrid models, and HF cards do not tell you enough. Both projects had one thing in common: when the output was wrong, I could *see* what was wrong. Garbled tokens. Wrong top-1 logit. A q/gate split that produced mean -1.449 instead of -0.033. The symptom pointed at the cause.

Diffusion models do not work that way.

**Wrong output is indistinguishable from other wrong output.** Four bugs in the MLX port each produced "noise." The mu schedule bug produced noise. The sigma endpoint bug produced noise. The timestep scale bug produced noise. The RoPE bug produced noise. I could not tell from the output image which component was broken, or even how many things were broken simultaneously. On a text model, each bug produces a distinct symptom. On a diffusion model, they all produce the same symptom.

**The scheduler is the model's contract with time.** Text inference has no scheduler. You run forward passes and sample tokens. Diffusion has a sigma schedule, a mu parameter, a timestep embedding, and a flow-match Euler step, and every one of them encodes assumptions about how the model relates each denoising step to the next. Three of four bugs were in this scheduling path. None of them exist in the text or video projects because those models do not denoise.

**You need a reference pipeline as an oracle.** On text, I can verify output by comparing tokens against a known answer. On video, I compared activations layer by layer against the Python reference. On diffusion, I had to build a full staged checkpoint comparison tool that dumps tensors at every pipeline boundary (text encoder output, VAE latent, per-step denoise latents) and compares them against the PyTorch MPS baseline using cosine similarity. Eyeballing an image tells you nothing. Cosine similarity at each gate tells you exactly where the first divergence occurs.

**Multiple bugs compound invisibly.** I could not find bugs 2-4 without fixing bug 1 first, because the wrong mu made the sigma schedule meaningless regardless of the endpoint. And I could not find the 4-bit precision issue without fixing all four bugs, because quantisation noise was indistinguishable from scheduling noise. On a text model, you can fix bugs in any order because each produces a different symptom. On a diffusion model, bugs stack and you have to peel them off one at a time, verifying each fix against the reference before moving to the next.

**Quantisation tolerance is task-dependent, not model-dependent.** Every text model I have tested runs at 4-bit. This decomposition model does not. A text model that confuses "the" with "a" is still readable. A decomposition model that miscalculates alpha by 10% produces visible artifacts in every pixel. The precision floor is set by the task, not the parameter count.

**The bandwidth-first methodology still transfers.** Different architecture, different modality, different output type. The 1000x gap between theoretical and observed throughput still pointed to the GgufLinear dequant overhead before I wrote any MLX code. Bandwidth analysis found the *speed* problem. The staged checkpoint comparison found the *correctness* problem. Both were necessary. Neither was sufficient alone.

## What is left

The engine achieves 458s total. Broken down:

| Stage | Time | % | What would help |
|---|---|---|---|
| Text encoder | 328s | 56% | Port to MLX (high complexity, high impact) |
| Denoise (8 steps) | ~100s | 34% | Reduce steps to 4-6 with quality validation |
| VAE decode | 30s | 8% | Decompose conv3d to per-frame conv2d on MPS |
| Other | 7s | 2% | — |

1. **Port text encoder to MLX** (estimated -40% total). The text encoder is Qwen2.5-VL-7B running fp32 on MPS. Porting to MLX 8-bit would apply the same fused quantised matmul that gave 2.3× on the DiT. Even a 2× speedup saves ~150s. Complexity: high. Requires implementing a full VLM pipeline.

2. **bf16 autocast for encode stages** (estimated -15% total). The text encoder runs fp32 because MlxBackend does not override MpsBackend's dtype. bf16 autocast was proven safe for denoise by the auto-researcher.

3. **Reduce denoise steps** (estimated -17% per step removed). The auto-researcher framework has the quality oracle to validate this automatically.

4. **Prompt caching** (estimated -56% for repeat prompts). The text encoder prompt is always "marketing asset." The conditioning could be cached to disk.
