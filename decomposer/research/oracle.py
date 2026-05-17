from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim


def composite_layers(layers: list[Image.Image]) -> Image.Image:
    if not layers:
        raise ValueError("composite_layers requires at least one layer")
    base = Image.new("RGBA", layers[0].size, (0, 0, 0, 0))
    for layer in layers:
        rgba = layer if layer.mode == "RGBA" else layer.convert("RGBA")
        base = Image.alpha_composite(base, rgba)
    return base


def composite_ssim(a: Image.Image, b: Image.Image) -> float:
    a_rgb = a.convert("RGB")
    b_rgb = b.convert("RGB")
    if a_rgb.size != b_rgb.size:
        b_rgb = b_rgb.resize(a_rgb.size, Image.BILINEAR)
    a_arr = np.asarray(a_rgb, dtype=np.float32) / 255.0
    b_arr = np.asarray(b_rgb, dtype=np.float32) / 255.0
    return float(ssim(a_arr, b_arr, channel_axis=2, data_range=1.0))


def per_layer_ssim_matched(
    experiment_layers: list[Image.Image],
    baseline_layers: list[Image.Image],
) -> tuple[float, list[int]]:
    if len(experiment_layers) != len(baseline_layers):
        raise ValueError(
            f"per_layer_ssim_matched requires same number of layers; "
            f"got experiment={len(experiment_layers)} baseline={len(baseline_layers)}"
        )
    n = len(experiment_layers)
    cost = np.zeros((n, n), dtype=np.float64)
    for i, exp in enumerate(experiment_layers):
        for j, base in enumerate(baseline_layers):
            cost[i, j] = -composite_ssim(exp, base)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = [-cost[i, col_ind[i]] for i in range(n)]
    matching = [int(col_ind[i]) for i in range(n)]
    return float(np.mean(matched)), matching


def check_non_degeneracy(
    layers: list[Image.Image],
    min_opaque_fraction: float = 0.01,
    min_transparent_fraction: float = 0.01,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for i, layer in enumerate(layers):
        rgba = layer if layer.mode == "RGBA" else layer.convert("RGBA")
        alpha = np.asarray(rgba.split()[-1])
        total = alpha.size
        opaque_frac = float((alpha > 0).sum()) / total
        transparent_frac = float((alpha < 255).sum()) / total
        if opaque_frac < min_opaque_fraction:
            reasons.append(
                f"layer {i}: opaque fraction {opaque_frac:.4f} < {min_opaque_fraction}"
            )
        if transparent_frac < min_transparent_fraction:
            reasons.append(
                f"layer {i}: transparent fraction {transparent_frac:.4f} < {min_transparent_fraction}"
            )
    return (len(reasons) == 0), reasons


def structural_similarity(
    experiment_layers: list[Image.Image],
    baseline_layers: list[Image.Image],
    matching: list[int],
) -> tuple[float, float]:
    alpha_diffs: list[float] = []
    rgb_dists: list[float] = []
    for i, exp_layer in enumerate(experiment_layers):
        base_layer = baseline_layers[matching[i]]
        exp_arr = np.asarray(exp_layer.convert("RGBA"), dtype=np.float32)
        base_arr = np.asarray(base_layer.convert("RGBA"), dtype=np.float32)
        if exp_arr.shape != base_arr.shape:
            base_layer_resized = base_layer.resize(exp_layer.size, Image.BILINEAR)
            base_arr = np.asarray(base_layer_resized.convert("RGBA"), dtype=np.float32)
        exp_alpha_cov = (exp_arr[..., 3] > 0).mean()
        base_alpha_cov = (base_arr[..., 3] > 0).mean()
        alpha_diffs.append(abs(exp_alpha_cov - base_alpha_cov))
        exp_rgb_mean = exp_arr[..., :3].mean(axis=(0, 1))
        base_rgb_mean = base_arr[..., :3].mean(axis=(0, 1))
        rgb_dists.append(float(np.linalg.norm(exp_rgb_mean - base_rgb_mean)))
    max_alpha_diff = float(max(alpha_diffs)) if alpha_diffs else 0.0
    max_rgb_dist = float(max(rgb_dists)) if rgb_dists else 0.0
    return max_alpha_diff, max_rgb_dist


@dataclass
class QualityReport:
    composite_ssim: float
    per_layer_ssim_matched: float
    per_layer_ssim_individual: list[float]
    layer_match_indices: list[int]
    non_degenerate: bool
    max_alpha_coverage_diff: float = 0.0
    max_rgb_mean_distance: float = 0.0
    degeneracy_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def passes(
        self,
        composite_ssim_min: float = 0.60,
        per_layer_ssim_min: float = 0.50,
        max_alpha_diff: float = 0.15,
        max_rgb_dist: float = 20.0,
    ) -> bool:
        return (
            self.non_degenerate
            and self.max_alpha_coverage_diff <= max_alpha_diff
            and self.max_rgb_mean_distance <= max_rgb_dist
            and self.composite_ssim >= composite_ssim_min
            and self.per_layer_ssim_matched >= per_layer_ssim_min
        )


def score(
    experiment_layers: list[Image.Image],
    baseline_layers: list[Image.Image],
    input_image: Image.Image,
) -> QualityReport:
    if not experiment_layers:
        return QualityReport(
            composite_ssim=0.0,
            per_layer_ssim_matched=0.0,
            per_layer_ssim_individual=[],
            layer_match_indices=[],
            non_degenerate=False,
            degeneracy_reasons=["experiment produced zero layers"],
        )

    composite = composite_layers(experiment_layers).convert("RGB")
    comp_ssim = composite_ssim(composite, input_image)

    matched_score, matching = per_layer_ssim_matched(experiment_layers, baseline_layers)
    individual = [
        composite_ssim(experiment_layers[i], baseline_layers[matching[i]])
        for i in range(len(experiment_layers))
    ]

    non_degen, reasons = check_non_degeneracy(experiment_layers)
    max_alpha_diff, max_rgb_dist = structural_similarity(
        experiment_layers, baseline_layers, matching,
    )
    return QualityReport(
        composite_ssim=comp_ssim,
        per_layer_ssim_matched=matched_score,
        per_layer_ssim_individual=individual,
        layer_match_indices=matching,
        non_degenerate=non_degen,
        max_alpha_coverage_diff=max_alpha_diff,
        max_rgb_mean_distance=max_rgb_dist,
        degeneracy_reasons=reasons,
    )
