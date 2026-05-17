from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


@dataclass
class QualityWarning:
    code: str
    message: str

def validate_layers(layers: list[Image.Image]) -> list[QualityWarning]:
    warnings: list[QualityWarning] = []

    coverages = []
    for i, layer in enumerate(layers):
        alpha = np.asarray(layer.convert("RGBA"))[..., 3]
        coverage = float((alpha > 0).sum()) / alpha.size
        coverages.append(coverage)
        if coverage == 0.0:
            warnings.append(QualityWarning(
                code="empty_layer",
                message=f"Layer {i} is fully transparent",
            ))

    has_partial = any(c < 0.90 for c in coverages)
    if not has_partial and len(layers) > 1:
        warnings.append(QualityWarning(
            code="all_opaque",
            message="All layers are >90% opaque, decomposition may have failed",
        ))

    if len(layers) >= 2:
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                a = np.asarray(layers[i].convert("RGB"), dtype=np.float32) / 255.0
                b = np.asarray(layers[j].convert("RGB"), dtype=np.float32) / 255.0
                if a.shape != b.shape:
                    b = np.array(layers[j].convert("RGB").resize(
                        layers[i].size, Image.BILINEAR
                    ), dtype=np.float32) / 255.0
                similarity = ssim(a, b, channel_axis=2, data_range=1.0)
                if similarity > 0.95:
                    warnings.append(QualityWarning(
                        code="low_diversity",
                        message=f"Layers {i} and {j} are near-identical (SSIM={similarity:.3f})",
                    ))

    return warnings
