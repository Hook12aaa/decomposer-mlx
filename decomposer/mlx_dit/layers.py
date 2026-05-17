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
    def __init__(self, num_channels: int, scale: float = 1000.0) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.scale = scale

    def __call__(self, t: mx.array) -> mx.array:
        t_scaled = t * self.scale
        half = self.num_channels // 2
        freqs = mx.exp(-math.log(10000.0) * mx.arange(0, half) / half)
        args = t_scaled[:, None] * freqs[None, :]
        return mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)


class MLXTimestepEmbedding(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.silu = nn.SiLU()
        self.linear2 = nn.Linear(hidden_dim, out_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(self.silu(self.linear1(x)))
