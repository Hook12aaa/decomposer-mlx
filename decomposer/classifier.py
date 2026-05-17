from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class LayerInfo:
    index: int
    classification: str
    confidence: float
    alpha_coverage: float
    bounding_box: tuple[int, int, int, int]
    mean_rgb: tuple[int, int, int]


def _compute_metrics(layer: Image.Image) -> dict:
    rgba = np.asarray(layer.convert("RGBA"), dtype=np.uint8)
    alpha = rgba[..., 3]
    h, w = alpha.shape
    total_pixels = h * w

    opaque_mask = alpha > 0
    alpha_coverage = float(opaque_mask.sum()) / total_pixels

    if alpha_coverage == 0.0:
        return {
            "alpha_coverage": 0.0,
            "bounding_box": (0, 0, 0, 0),
            "mean_rgb": (0, 0, 0),
            "area_fraction": 0.0,
            "position_y_center": 0.5,
        }

    rows = np.any(opaque_mask, axis=1)
    cols = np.any(opaque_mask, axis=0)
    y0, y1 = int(np.argmax(rows)), int(h - np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), int(w - np.argmax(cols[::-1]))

    bbox_area = (x1 - x0) * (y1 - y0)
    area_fraction = bbox_area / total_pixels

    rgb_opaque = rgba[opaque_mask][:, :3]
    mean_rgb = tuple(int(c) for c in rgb_opaque.mean(axis=0))

    position_y_center = ((y0 + y1) / 2) / h

    return {
        "alpha_coverage": alpha_coverage,
        "bounding_box": (x0, y0, x1, y1),
        "mean_rgb": mean_rgb,
        "area_fraction": area_fraction,
        "position_y_center": position_y_center,
    }


def _classify_single(metrics: dict) -> tuple[str, float]:
    cov = metrics["alpha_coverage"]
    area = metrics["area_fraction"]

    if cov == 0.0:
        return "unknown", 0.0

    if cov > 0.95:
        return "background", min(1.0, cov)

    if area < 0.15 and cov < 0.15:
        return "logo", 0.7 + 0.3 * (1.0 - area / 0.15)

    if area < 0.10 and metrics["position_y_center"] > 0.5:
        return "cta_button", 0.6

    if 0.05 <= cov <= 0.30 and area > 0.30:
        return "overlay", 0.6

    if 0.10 <= cov <= 0.85 and area > 0.15:
        return "hero_image", 0.5 + 0.3 * min(1.0, area / 0.5)

    if cov < 0.10:
        return "text", 0.5

    return "unknown", 0.3


def classify_layers(layers: list[Image.Image]) -> list[LayerInfo]:
    results: list[LayerInfo] = []
    for i, layer in enumerate(layers):
        metrics = _compute_metrics(layer)
        classification, confidence = _classify_single(metrics)
        results.append(LayerInfo(
            index=i,
            classification=classification,
            confidence=confidence,
            alpha_coverage=metrics["alpha_coverage"],
            bounding_box=metrics["bounding_box"],
            mean_rgb=metrics["mean_rgb"],
        ))
    return results
