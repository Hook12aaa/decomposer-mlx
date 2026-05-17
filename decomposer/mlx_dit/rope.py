from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx import nn


class MLXRoPE3D(nn.Module):
    def __init__(self, axes_dims: list[int], theta: float = 10000.0, scale_rope: bool = True) -> None:
        super().__init__()
        self.axes_dims = axes_dims
        self.theta = theta
        self.scale_rope = scale_rope

        max_seq = 4096
        pos_index = np.arange(max_seq, dtype=np.float32)
        neg_index = np.arange(max_seq, dtype=np.float32)[::-1] * -1.0 - 1.0

        self.pos_freqs = np.concatenate(
            [self._rope_params(pos_index, dim) for dim in self.axes_dims],
            axis=1,
        )
        self.neg_freqs = np.concatenate(
            [self._rope_params(neg_index, dim) for dim in self.axes_dims],
            axis=1,
        )

    def _rope_params(self, index: np.ndarray, dim: int) -> np.ndarray:
        scales = np.arange(0, dim, 2, dtype=np.float32) / dim
        omega = 1.0 / (self.theta ** scales)
        freqs = np.outer(index.astype(np.float32), omega)
        cos_freqs = np.cos(freqs)
        sin_freqs = np.sin(freqs)
        return np.stack([cos_freqs, sin_freqs], axis=-1)

    def _split_freqs(self, freqs: np.ndarray) -> list[np.ndarray]:
        axes_splits = [x // 2 for x in self.axes_dims]
        return np.split(freqs, np.cumsum(axes_splits)[:-1], axis=1)

    def _spatial_freqs(self, dim_freqs_pos: np.ndarray, dim_freqs_neg: np.ndarray, size: int) -> np.ndarray:
        if self.scale_rope:
            head = size // 2
            tail = size - head
            return np.concatenate([dim_freqs_neg[-tail:], dim_freqs_pos[:head]], axis=0)
        return dim_freqs_pos[:size]

    def compute_video_freqs(
        self, frame: int, height: int, width: int, layer_idx: int
    ) -> np.ndarray:
        pos_split = self._split_freqs(self.pos_freqs)
        neg_split = self._split_freqs(self.neg_freqs)

        freqs_frame = pos_split[0][layer_idx:layer_idx + frame].reshape(frame, 1, 1, -1, 2)
        freqs_frame = np.broadcast_to(freqs_frame, (frame, height, width, freqs_frame.shape[-2], 2))

        freqs_height = self._spatial_freqs(pos_split[1], neg_split[1], height).reshape(1, height, 1, -1, 2)
        freqs_height = np.broadcast_to(freqs_height, (frame, height, width, freqs_height.shape[-2], 2))

        freqs_width = self._spatial_freqs(pos_split[2], neg_split[2], width).reshape(1, 1, width, -1, 2)
        freqs_width = np.broadcast_to(freqs_width, (frame, height, width, freqs_width.shape[-2], 2))

        freqs = np.concatenate([freqs_frame, freqs_height, freqs_width], axis=-2)
        seq_len = frame * height * width
        return freqs.reshape(seq_len, -1, 2).copy()

    def compute_condition_freqs(
        self, frame: int, height: int, width: int
    ) -> np.ndarray:
        pos_split = self._split_freqs(self.pos_freqs)
        neg_split = self._split_freqs(self.neg_freqs)

        freqs_frame = neg_split[0][-1:].reshape(frame, 1, 1, -1, 2)
        freqs_frame = np.broadcast_to(freqs_frame, (frame, height, width, freqs_frame.shape[-2], 2))

        freqs_height = self._spatial_freqs(pos_split[1], neg_split[1], height).reshape(1, height, 1, -1, 2)
        freqs_height = np.broadcast_to(freqs_height, (frame, height, width, freqs_height.shape[-2], 2))

        freqs_width = self._spatial_freqs(pos_split[2], neg_split[2], width).reshape(1, 1, width, -1, 2)
        freqs_width = np.broadcast_to(freqs_width, (frame, height, width, freqs_width.shape[-2], 2))

        freqs = np.concatenate([freqs_frame, freqs_height, freqs_width], axis=-2)
        seq_len = frame * height * width
        return freqs.reshape(seq_len, -1, 2).copy()

    def compute_freqs(
        self, height: int, width: int, num_frames: int
    ) -> tuple[mx.array, mx.array]:
        num_layers = num_frames - 1
        vid_freqs = []
        for idx in range(num_layers):
            vid_freqs.append(self.compute_video_freqs(1, height, width, idx))
        vid_freqs.append(self.compute_condition_freqs(1, height, width))
        all_vid = np.concatenate(vid_freqs, axis=0)

        cos_freqs = mx.array(all_vid[..., 0].astype(np.float32))
        sin_freqs = mx.array(all_vid[..., 1].astype(np.float32))

        max_vid_index = max(height // 2, width // 2, num_layers) if self.scale_rope else max(height, width, num_layers)
        return cos_freqs, sin_freqs, max_vid_index
