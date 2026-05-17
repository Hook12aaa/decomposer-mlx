"""MlxBackend: runs the denoise loop in MLX, delegates text encoder + VAE to MpsBackend."""

from __future__ import annotations

import gc
import logging
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image

from decomposer.config import Settings, get_settings
from decomposer.core.mps_backend import MpsBackend
from decomposer.core.xray import Tracer

logger = logging.getLogger(__name__)


def _pack_latents_np(
    latents: np.ndarray,
    batch_size: int,
    num_channels: int,
    height: int,
    width: int,
    num_frames: int,
) -> np.ndarray:
    """Pack latents from (B, F, C, H, W) to (B, F*H/2*W/2, C*4) for transformer input."""
    latents = latents.reshape(batch_size, num_frames, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.transpose(0, 1, 3, 5, 2, 4, 6)
    latents = latents.reshape(batch_size, num_frames * (height // 2) * (width // 2), num_channels * 4)
    return latents


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
        logger.info(
            "mlx decompose start run_id=%s layers=%d steps=%d",
            t.run_id, layers, steps,
        )

        cond = self._mps._encode_prompt(image, prompt="marketing asset", tracer=t)
        image_latent, height, width = self._mps._encode_image_to_latent(
            image, resolution=resolution, tracer=t
        )

        with t.stage("load_dit"):
            from decomposer.mlx_dit.loader import load_mlx_transformer
            model = load_mlx_transformer(self.settings.mlx_weights_dir)
            model.__call__ = mx.compile(model.__call__)

        with t.stage(
            "denoise_loop", steps=steps, layers=layers, resolution=resolution
        ):
            result_latents = self._denoise_mlx(
                model=model,
                cond=cond,
                image_latent=image_latent,
                height=height,
                width=width,
                layers=layers,
                steps=steps,
                seed=seed,
                tracer=t,
            )

        with t.stage("free_dit"):
            del model
            gc.collect()
            mx.clear_cache()

        original_device = self._mps.device
        self._mps.device = "cpu"
        self._mps.residency.device = "cpu"
        try:
            return self._mps._decode(
                result_latents, layers=layers, height=height, width=width, tracer=t
            )
        finally:
            self._mps.device = original_device
            self._mps.residency.device = original_device

    def _denoise_mlx(
        self,
        model,
        cond: torch.Tensor,
        image_latent: torch.Tensor,
        height: int,
        width: int,
        layers: int,
        steps: int,
        seed: int | None,
        tracer: Tracer,
    ) -> torch.Tensor:
        import mlx.core as mx
        from decomposer.mlx_dit.scheduler import (
            compute_mu,
            flow_match_euler_step,
            get_sigmas,
            get_timesteps,
        )

        vae_scale_factor = 8
        num_frames = layers + 1
        num_channels = 16
        lat_h = 2 * (height // (vae_scale_factor * 2))
        lat_w = 2 * (width // (vae_scale_factor * 2))

        if seed is not None:
            mx.random.seed(seed)

        noise_shape = (1, num_frames, num_channels, lat_h, lat_w)
        noise_np = np.random.default_rng(seed).standard_normal(noise_shape).astype(np.float32)
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

        sigmas = get_sigmas(steps, mu=mu)
        timesteps_list = get_timesteps(sigmas)

        rope_h = lat_h // 2
        rope_w = lat_w // 2
        patches_per_frame = rope_h * rope_w

        for i in range(steps):
            with tracer.step("denoise_step", i=i):
                sigma = sigmas[i]
                sigma_next = sigmas[i + 1]
                timestep_val = timesteps_list[i] / 1000.0

                latent_model_input = mx.concatenate(
                    [latents, image_latent_mlx], axis=1
                )

                actual_frames = latent_model_input.shape[1] // patches_per_frame
                timestep_mlx = mx.array([timestep_val], dtype=latents.dtype)

                noise_pred = model(
                    latent_model_input,
                    cond_mlx,
                    timestep_mlx,
                    height=rope_h,
                    width=rope_w,
                    num_frames=actual_frames,
                )
                mx.eval(noise_pred)

                noise_pred = noise_pred[:, :latents.shape[1], :]

                latents = flow_match_euler_step(
                    latents, noise_pred, sigma, sigma_next
                )
                mx.eval(latents)

        latents_np = np.array(latents)
        result = torch.from_numpy(latents_np)
        if torch.isnan(result).any():
            raise RuntimeError(
                f"NaN detected in latents after {steps} denoise steps. "
                "model output diverged. Try a different seed or reduce resolution."
            )
        return result
