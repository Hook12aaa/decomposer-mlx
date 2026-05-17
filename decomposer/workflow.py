from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from PIL import Image

from decomposer.classifier import classify_layers
from decomposer.config import Settings, get_settings
from decomposer.core.backend import get_backend
from decomposer.core.perfetto import report_to_json
from decomposer.core.xray import Tracer
from decomposer.manifest import LayerEntry, Manifest, write_manifest
from decomposer.preprocess import PreprocessError, preprocess
from decomposer.quality import validate_layers

logger = logging.getLogger(__name__)

MINIMUM_MEMORY_GB = 20


class PreflightError(Exception):
    pass


@dataclass
class WorkflowResult:
    success: bool
    layer_files: list[Path] = field(default_factory=list)
    manifest_path: Path = Path()
    quality_warnings: list[str] = field(default_factory=list)
    wall_time_seconds: float = 0.0
    error: str = ""


def _preflight(backend_name: str, settings: Settings) -> None:
    if backend_name == "mlx":
        weights_dir = settings.mlx_weights_dir
        if not weights_dir.exists() or not any(weights_dir.glob("*.safetensors")):
            raise PreflightError(
                f"MLX weights not found at {weights_dir}. "
                "Run `decomposer convert-to-mlx <gguf_path>` first."
            )

    if backend_name in ("mlx", "mps"):
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        if available_gb < MINIMUM_MEMORY_GB:
            logger.warning(
                "Available memory %.1f GB is below recommended %d GB, may OOM",
                available_gb, MINIMUM_MEMORY_GB,
            )


def run_workflow(
    image_path: Path,
    output_dir: Path,
    layers: int,
    resolution: int,
    steps: int,
    seed: int | None = None,
    backend_name: str = "mlx",
    trace: bool = False,
) -> WorkflowResult:
    start = time.perf_counter()
    settings = get_settings()

    if not image_path.exists():
        return WorkflowResult(success=False, error=f"Image not found: {image_path}")

    try:
        raw_image = Image.open(image_path)
    except Exception as e:
        return WorkflowResult(success=False, error=f"Cannot open image: {e}")

    source_dimensions = raw_image.size

    try:
        image = preprocess(raw_image)
    except PreprocessError as e:
        return WorkflowResult(success=False, error=str(e))

    try:
        _preflight(backend_name, settings)
    except PreflightError as e:
        return WorkflowResult(success=False, error=str(e))

    output_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(run_id=f"run-{int(time.time())}")
    backend = get_backend(backend_name)

    try:
        result_layers = backend.decompose(
            image, layers=layers, resolution=resolution,
            steps=steps, seed=seed, tracer=tracer,
        )
    except Exception as e:
        return WorkflowResult(success=False, error=f"Inference failed: {e}")

    quality_warns = validate_layers(result_layers)
    warning_messages = [w.message for w in quality_warns]

    layer_files: list[Path] = []
    for i, layer_img in enumerate(result_layers):
        path = output_dir / f"layer_{i}.png"
        layer_img.save(path)
        layer_files.append(path)

    infos = classify_layers(result_layers)
    layer_entries = [
        LayerEntry.from_layer_info(
            info,
            file=f"layer_{info.index}.png",
            file_size_bytes=layer_files[info.index].stat().st_size,
        )
        for info in infos
    ]

    wall_time = time.perf_counter() - start
    manifest = Manifest(
        source=image_path.name,
        source_dimensions=source_dimensions,
        resolution=resolution,
        steps=steps,
        layers_requested=layers,
        backend=backend_name,
        seed=seed,
        wall_time_seconds=round(wall_time, 2),
        quality_warnings=warning_messages,
        layers=layer_entries,
    )
    manifest_path = output_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    report = tracer.report()
    if trace:
        (output_dir / "trace.json").write_text(report_to_json(report))

    return WorkflowResult(
        success=True,
        layer_files=layer_files,
        manifest_path=manifest_path,
        quality_warnings=warning_messages,
        wall_time_seconds=round(wall_time, 2),
    )
