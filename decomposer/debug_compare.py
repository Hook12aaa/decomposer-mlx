"""Side-by-side MLX vs MPS pipeline comparison with per-stage artifact dumps.

Runs both backends on the same image with the same seed, dumps intermediate
tensors at each checkpoint, and reports where they diverge.
"""
from __future__ import annotations

import gc
import json
import os
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch
from pathlib import Path
from PIL import Image

from decomposer.config import get_settings
from decomposer.core.mps_backend import MpsBackend
from decomposer.core.xray import Tracer
from decomposer.core.mlx_backend import _pack_latents_np


def dump_tensor(t, name: str, out_dir: Path):
    if isinstance(t, torch.Tensor):
        arr = t.float().cpu().numpy()
    else:
        arr = np.array(t)
    stats = {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "nan_count": int(np.isnan(arr).sum()),
        "abs_mean": float(np.abs(arr).mean()),
    }
    np.save(out_dir / f"{name}.npy", arr)
    return stats


def run_mps_staged(image: Image.Image, out_dir: Path, seed: int = 42):
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    backend = MpsBackend(settings=settings)
    tracer = Tracer(run_id="mps-debug")
    report = {}

    cond = backend._encode_prompt(image, prompt="marketing asset", tracer=tracer)
    report["cond"] = dump_tensor(cond, "cond", out_dir)
    print(f"  MPS cond: {report['cond']}")

    image_latent, height, width = backend._encode_image_to_latent(
        image, resolution=640, tracer=tracer
    )
    report["image_latent"] = dump_tensor(image_latent, "image_latent", out_dir)
    report["height"] = height
    report["width"] = width
    print(f"  MPS image_latent: {report['image_latent']}")

    from diffusers import FlowMatchEulerDiscreteScheduler
    from decomposer.core.gguf_pipeline import load_qwen_image_layered_transformer_q8
    from huggingface_hub import hf_hub_download

    gguf_path = hf_hub_download(
        repo_id=settings.gguf_repo, filename=settings.gguf_file
    )
    dit = backend.residency.load(
        "dit",
        lambda: load_qwen_image_layered_transformer_q8(
            gguf_path, dtype=backend.dtype,
            expected_sha256=settings.gguf_sha256,
        ),
    )
    dit.eval()

    pipe = backend._build_pipeline(dit, vae=None)

    precomputed_latent = image_latent.to(device=backend.device, dtype=backend.dtype)
    pipe._encode_vae_image = lambda image=None, generator=None: precomputed_latent

    prompt_embeds = cond.to(device=backend.device, dtype=backend.dtype)
    if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)

    generator = torch.Generator(device=backend.device).manual_seed(seed)

    step_latents = []

    def _capture_cb(pipe_self, step_i, t, cbk):
        latents = cbk.get("latents")
        if latents is not None:
            step_latents.append(latents.detach().cpu().clone())
        return cbk

    with torch.autocast(device_type='mps', dtype=torch.bfloat16):
        final_latents = pipe.denoise_only(
            image=image,
            prompt="decompose",
            prompt_embeds=prompt_embeds,
            layers=3,
            num_inference_steps=8,
            resolution=640,
            generator=generator,
            callback_on_step_end=_capture_cb,
        )

    for i, sl in enumerate(step_latents):
        report[f"step_{i}"] = dump_tensor(sl, f"step_{i}", out_dir)
        print(f"  MPS step {i}: mean={report[f'step_{i}']['mean']:.6f} std={report[f'step_{i}']['std']:.6f}")

    report["final_latents"] = dump_tensor(final_latents, "final_latents", out_dir)
    print(f"  MPS final: {report['final_latents']}")

    del pipe, dit
    backend.residency.free()
    gc.collect()
    torch.mps.empty_cache()

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def run_mlx_staged(image: Image.Image, out_dir: Path, seed: int = 42,
                   mps_cond=None, mps_image_latent=None, mps_height=None, mps_width=None):
    import mlx.core as mx
    from decomposer.mlx_dit.loader import load_mlx_transformer
    from decomposer.mlx_dit.scheduler import compute_mu, flow_match_euler_step, get_sigmas, get_timesteps

    out_dir.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    report = {}

    if mps_cond is not None:
        cond = mps_cond
        image_latent = mps_image_latent
        height, width = mps_height, mps_width
        print("  MLX: reusing MPS cond + image_latent (shared stages)")
    else:
        backend = MpsBackend(settings=settings)
        tracer = Tracer(run_id="mlx-debug-encode")
        cond = backend._encode_prompt(image, prompt="marketing asset", tracer=tracer)
        image_latent, height, width = backend._encode_image_to_latent(
            image, resolution=640, tracer=tracer
        )

    report["cond"] = dump_tensor(cond, "cond", out_dir)
    report["image_latent"] = dump_tensor(image_latent, "image_latent", out_dir)
    report["height"] = height
    report["width"] = width

    model = load_mlx_transformer(settings.mlx_weights_dir)

    vae_scale_factor = 8
    layers = 3
    num_frames = layers + 1
    num_channels = 16
    lat_h = 2 * (height // (vae_scale_factor * 2))
    lat_w = 2 * (width // (vae_scale_factor * 2))

    noise_np = np.random.default_rng(seed).standard_normal(
        (1, num_frames, num_channels, lat_h, lat_w)
    ).astype(np.float32)

    report["noise_raw"] = dump_tensor(
        torch.from_numpy(noise_np), "noise_raw", out_dir
    )

    noise_packed = _pack_latents_np(noise_np, 1, num_channels, lat_h, lat_w, num_frames)
    latents = mx.array(noise_packed)

    image_latent_np = image_latent.float().numpy()
    image_latent_packed = _pack_latents_np(
        image_latent_np.transpose(0, 2, 1, 3, 4),
        1, num_channels,
        image_latent_np.shape[3], image_latent_np.shape[4],
        1,
    )
    image_latent_mlx = mx.array(image_latent_packed)

    cond_mlx = mx.array(cond.float().numpy())
    if cond_mlx.ndim == 2:
        cond_mlx = cond_mlx[None, :, :]

    image_seq_len = image_latent_mlx.shape[1]
    mu = compute_mu(image_seq_len)
    sigmas = get_sigmas(8, mu=mu)
    timesteps = get_timesteps(sigmas)

    report["mu"] = mu
    report["sigmas"] = sigmas

    rope_h = lat_h // 2
    rope_w = lat_w // 2
    patches_per_frame = rope_h * rope_w

    print(f"  MLX mu={mu:.4f}, sigmas={[f'{s:.4f}' for s in sigmas[:3]]}...{sigmas[-2]:.4f}")

    for i in range(8):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        timestep_val = timesteps[i] / 1000.0

        latent_model_input = mx.concatenate([latents, image_latent_mlx], axis=1)
        actual_frames = latent_model_input.shape[1] // patches_per_frame
        timestep_mlx = mx.array([timestep_val], dtype=latents.dtype)

        noise_pred = model(
            latent_model_input, cond_mlx, timestep_mlx,
            height=rope_h, width=rope_w, num_frames=actual_frames,
        )
        mx.eval(noise_pred)

        noise_pred_sliced = noise_pred[:, :latents.shape[1], :]

        report[f"noise_pred_{i}"] = dump_tensor(
            torch.from_numpy(np.array(noise_pred_sliced)),
            f"noise_pred_{i}", out_dir,
        )

        latents = flow_match_euler_step(latents, noise_pred_sliced, sigma, sigma_next)
        mx.eval(latents)

        report[f"step_{i}"] = dump_tensor(
            torch.from_numpy(np.array(latents)),
            f"step_{i}", out_dir,
        )
        print(f"  MLX step {i}: mean={report[f'step_{i}']['mean']:.6f} std={report[f'step_{i}']['std']:.6f}")

    report["final_latents"] = dump_tensor(
        torch.from_numpy(np.array(latents)), "final_latents", out_dir
    )
    print(f"  MLX final: {report['final_latents']}")

    del model
    gc.collect()

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def compare_reports(mps_report: dict, mlx_report: dict):
    print("\n" + "=" * 70)
    print("CHECKPOINT COMPARISON: MPS vs MLX")
    print("=" * 70)

    shared_keys = ["cond", "image_latent"]
    for key in shared_keys:
        if key in mps_report and key in mlx_report:
            m = mps_report[key]
            x = mlx_report[key]
            print(f"\n{key}:")
            print(f"  MPS: mean={m['mean']:.6f} std={m['std']:.6f} shape={m['shape']}")
            print(f"  MLX: mean={x['mean']:.6f} std={x['std']:.6f} shape={x['shape']}")
            if m['shape'] == x['shape']:
                print(f"  MATCH: shapes identical")
            else:
                print(f"  MISMATCH: shapes differ!")

    print(f"\n{'─' * 70}")
    print(f"{'Step':<8} {'MPS mean':>12} {'MLX mean':>12} {'MPS std':>12} {'MLX std':>12} {'Verdict':>10}")
    print(f"{'─' * 70}")

    for i in range(8):
        key = f"step_{i}"
        if key in mps_report and key in mlx_report:
            m = mps_report[key]
            x = mlx_report[key]
            mean_diff = abs(m['mean'] - x['mean'])
            std_ratio = x['std'] / m['std'] if m['std'] > 0 else float('inf')
            verdict = "OK" if mean_diff < 0.5 and 0.5 < std_ratio < 2.0 else "DIVERGED"
            print(f"{key:<8} {m['mean']:>12.6f} {x['mean']:>12.6f} {m['std']:>12.6f} {x['std']:>12.6f} {verdict:>10}")

    m_final = mps_report.get("final_latents", {})
    x_final = mlx_report.get("final_latents", {})
    if m_final and x_final:
        print(f"\nFinal latents:")
        print(f"  MPS: mean={m_final['mean']:.6f} std={m_final['std']:.6f} range=[{m_final['min']:.4f}, {m_final['max']:.4f}]")
        print(f"  MLX: mean={x_final['mean']:.6f} std={x_final['std']:.6f} range=[{x_final['min']:.4f}, {x_final['max']:.4f}]")


if __name__ == "__main__":
    image = Image.open("test_image.jpg").convert("RGBA")
    base = Path("out/debug-compare")

    print("=" * 70)
    print("STAGE 1: MPS Pipeline (ground truth)")
    print("=" * 70)
    mps_report = run_mps_staged(image, base / "mps", seed=42)

    mps_cond = torch.from_numpy(np.load(base / "mps" / "cond.npy"))
    mps_image_latent = torch.from_numpy(np.load(base / "mps" / "image_latent.npy"))

    print("\n" + "=" * 70)
    print("STAGE 2: MLX Pipeline (same cond + image_latent)")
    print("=" * 70)
    mlx_report = run_mlx_staged(
        image, base / "mlx", seed=42,
        mps_cond=mps_cond,
        mps_image_latent=mps_image_latent,
        mps_height=mps_report["height"],
        mps_width=mps_report["width"],
    )

    compare_reports(mps_report, mlx_report)
