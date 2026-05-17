from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class Baseline:
    run_id: str
    input_image: Image.Image
    layers: list[Image.Image]
    trace: dict[str, Any]
    layers_count: int
    resolution: int
    steps: int
    commit_sha: str
    extras: dict[str, Any] = field(default_factory=dict)


def save_baseline(baseline: Baseline, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline.input_image.save(out_dir / "input.png")
    for i, layer in enumerate(baseline.layers):
        layer.save(out_dir / f"layer_{i}.png")
    (out_dir / "trace.json").write_text(json.dumps(baseline.trace, indent=2))
    meta = {
        "run_id": baseline.run_id,
        "layers_count": baseline.layers_count,
        "resolution": baseline.resolution,
        "steps": baseline.steps,
        "commit_sha": baseline.commit_sha,
        "extras": baseline.extras,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def load_baseline(out_dir: Path) -> Baseline:
    meta = json.loads((out_dir / "meta.json").read_text())
    trace = json.loads((out_dir / "trace.json").read_text())
    input_image = Image.open(out_dir / "input.png").convert("RGB")
    layers: list[Image.Image] = []
    for i in range(meta["layers_count"]):
        layers.append(Image.open(out_dir / f"layer_{i}.png").convert("RGBA"))
    return Baseline(
        run_id=meta["run_id"],
        input_image=input_image,
        layers=layers,
        trace=trace,
        layers_count=meta["layers_count"],
        resolution=meta["resolution"],
        steps=meta["steps"],
        commit_sha=meta["commit_sha"],
        extras=meta.get("extras", {}),
    )
