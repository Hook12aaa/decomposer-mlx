from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from decomposer.classifier import LayerInfo


class LayerEntry(BaseModel):
    file: str
    index: int
    classification: str
    confidence: float
    alpha_coverage: float
    bounding_box: tuple[int, int, int, int]
    mean_rgb: tuple[int, int, int]
    file_size_bytes: int

    @classmethod
    def from_layer_info(
        cls, info: LayerInfo, file: str, file_size_bytes: int
    ) -> LayerEntry:
        return cls(
            file=file,
            index=info.index,
            classification=info.classification,
            confidence=info.confidence,
            alpha_coverage=info.alpha_coverage,
            bounding_box=info.bounding_box,
            mean_rgb=info.mean_rgb,
            file_size_bytes=file_size_bytes,
        )


class Manifest(BaseModel):
    source: str
    source_dimensions: tuple[int, int]
    resolution: int
    steps: int
    layers_requested: int
    backend: str
    seed: int | None
    wall_time_seconds: float
    quality_warnings: list[str]
    layers: list[LayerEntry]


def write_manifest(manifest: Manifest, path: Path) -> None:
    path.write_text(manifest.model_dump_json(indent=2))
