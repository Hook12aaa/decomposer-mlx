"""FlowMatch Euler scheduler for the MLX denoise loop.

Implements the same sigma schedule as diffusers'
``FlowMatchEulerDiscreteScheduler`` with ``use_dynamic_shifting=True``
and ``time_shift_type='linear'``.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def compute_mu(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 8192,
    base_shift: float = 0.5,
    max_shift: float = 0.9,
) -> float:
    """Compute the dynamic shift mu based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return float(m * image_seq_len + b)


def get_sigmas(
    num_steps: int,
    mu: float,
    num_train_timesteps: int = 1000,
) -> list[float]:
    """Compute the sigma schedule for FlowMatch Euler discrete sampling.

    Returns a list of ``num_steps + 1`` sigma values (the last is 0.0).
    """
    sigmas_np = np.linspace(1.0, 1.0 / num_train_timesteps, num_steps).astype(np.float64)
    shifted = np.exp(mu) / (np.exp(mu) + ((1.0 / sigmas_np - 1.0) ** 1.0))
    sigmas = shifted.tolist() + [0.0]
    return sigmas


def get_timesteps(sigmas: list[float], num_train_timesteps: int = 1000) -> list[float]:
    """Convert sigmas to timestep values (sigma * num_train_timesteps)."""
    return [s * num_train_timesteps for s in sigmas[:-1]]


def flow_match_euler_step(
    latent: mx.array,
    noise_pred: mx.array,
    sigma: float,
    sigma_next: float,
) -> mx.array:
    """Single Euler step: latent + dt * noise_pred."""
    dt = mx.array(sigma_next - sigma, dtype=latent.dtype)
    return latent + dt * noise_pred
