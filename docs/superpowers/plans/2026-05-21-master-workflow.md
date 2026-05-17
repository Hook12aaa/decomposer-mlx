# Master Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single `decomposer run image.jpg` command that reliably decomposes any marketing image into classified RGBA layers with metadata JSON, using the MLX 4-bit backend on M3 Max.

**Architecture:** Five-stage pipeline: preprocess → preflight → MLX inference → classify → export. New modules for preprocessing, classification, manifest writing, and a workflow orchestrator that chains them. The existing `MlxBackend` is used as-is for inference, with `mx.compile` applied for performance. A new `run` CLI command wraps the workflow with smart defaults.

**Tech Stack:** Python 3.12+, MLX, PyTorch (text encoder/VAE only), PIL, pydantic, typer, numpy, scikit-image

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `decomposer/preprocess.py` | Create | Input validation, RGBA conversion, dimension checks |
| `tests/test_preprocess.py` | Create | Tests for preprocessor |
| `decomposer/classifier.py` | Create | Heuristic layer classification (alpha, bbox, position, color) |
| `tests/test_classifier.py` | Create | Tests for classifier with synthetic RGBA layers |
| `decomposer/manifest.py` | Create | Pydantic manifest schema + JSON writer |
| `tests/test_manifest.py` | Create | Tests for manifest serialization |
| `decomposer/quality.py` | Create | Post-decomposition quality validation |
| `tests/test_quality.py` | Create | Tests for quality checks |
| `decomposer/workflow.py` | Create | Orchestrator: preprocess → preflight → inference → classify → export |
| `tests/test_workflow.py` | Create | Tests for workflow with FakeBackend |
| `decomposer/cli.py` | Modify | Add `run` command |
| `tests/test_cli.py` | Modify | Add tests for `run` command |
| `decomposer/core/mlx_backend.py` | Modify | Apply `mx.compile` to denoise loop |

---

### Task 1: Input Preprocessor

**Files:**
- Create: `tests/test_preprocess.py`
- Create: `decomposer/preprocess.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_preprocess.py
import pytest
from PIL import Image

from decomposer.preprocess import preprocess, PreprocessError


def test_rgb_image_converted_to_rgba():
    img = Image.new("RGB", (256, 256), (100, 150, 200))
    result = preprocess(img)
    assert result.mode == "RGBA"
    assert result.size == (256, 256)


def test_rgba_image_passes_through():
    img = Image.new("RGBA", (512, 512), (100, 150, 200, 255))
    result = preprocess(img)
    assert result.mode == "RGBA"
    assert result.size == (512, 512)


def test_grayscale_converted_to_rgba():
    img = Image.new("L", (128, 128), 128)
    result = preprocess(img)
    assert result.mode == "RGBA"


def test_image_too_small_raises():
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    with pytest.raises(PreprocessError, match="64"):
        preprocess(img)


def test_image_exactly_64x64_passes():
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    result = preprocess(img)
    assert result.mode == "RGBA"


def test_oversized_image_logs_warning(caplog):
    img = Image.new("RGB", (4096, 2048), (0, 0, 0))
    import logging
    with caplog.at_level(logging.WARNING, logger="decomposer.preprocess"):
        result = preprocess(img)
    assert result.mode == "RGBA"
    assert any("2048" in r.message or "4096" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decomposer.preprocess'`

- [ ] **Step 3: Write minimal implementation**

```python
# decomposer/preprocess.py
from __future__ import annotations

import logging

from PIL import Image

logger = logging.getLogger(__name__)

MIN_DIMENSION = 64
WARN_DIMENSION = 2048


class PreprocessError(Exception):
    pass


def preprocess(image: Image.Image) -> Image.Image:
    w, h = image.size
    if w < MIN_DIMENSION or h < MIN_DIMENSION:
        raise PreprocessError(
            f"Image too small ({w}x{h}). Minimum dimension is {MIN_DIMENSION}px."
        )
    if w > WARN_DIMENSION or h > WARN_DIMENSION:
        logger.warning(
            "Image is %dx%d — will be downscaled to inference resolution", w, h
        )
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    return image
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_preprocess.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add decomposer/preprocess.py tests/test_preprocess.py
git commit -m "feat: add input preprocessor with validation"
```

---

### Task 2: Heuristic Layer Classifier

**Files:**
- Create: `tests/test_classifier.py`
- Create: `decomposer/classifier.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_classifier.py
import numpy as np
import pytest
from PIL import Image

from decomposer.classifier import classify_layers, LayerInfo


def _make_layer(w: int, h: int, alpha_fill: int = 255, color: tuple = (100, 100, 100)) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = color
    arr[..., 3] = alpha_fill
    return Image.fromarray(arr, "RGBA")


def _make_partial_layer(
    w: int, h: int, x0: int, y0: int, x1: int, y1: int,
    color: tuple = (200, 50, 50), alpha: int = 255,
) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[y0:y1, x0:x1, :3] = color
    arr[y0:y1, x0:x1, 3] = alpha
    return Image.fromarray(arr, "RGBA")


def test_full_opaque_layer_classified_as_background():
    layer = _make_layer(640, 480, alpha_fill=255)
    infos = classify_layers([layer])
    assert len(infos) == 1
    assert infos[0].classification == "background"


def test_empty_layer_has_zero_coverage():
    layer = _make_layer(640, 480, alpha_fill=0)
    infos = classify_layers([layer])
    assert infos[0].alpha_coverage == 0.0


def test_partial_layer_has_correct_bounding_box():
    layer = _make_partial_layer(640, 480, 100, 50, 300, 200)
    infos = classify_layers([layer])
    assert infos[0].bounding_box == (100, 50, 300, 200)


def test_small_compact_element_classified_as_logo():
    bg = _make_layer(640, 480, alpha_fill=255)
    logo = _make_partial_layer(640, 480, 10, 10, 70, 50, color=(255, 0, 0))
    infos = classify_layers([bg, logo])
    assert infos[1].classification == "logo"


def test_mid_coverage_large_area_classified_as_hero():
    bg = _make_layer(640, 480, alpha_fill=255)
    hero = _make_partial_layer(640, 480, 50, 50, 500, 400, color=(100, 200, 150))
    infos = classify_layers([bg, hero])
    assert infos[1].classification == "hero_image"


def test_classify_returns_correct_mean_rgb():
    layer = _make_layer(100, 100, alpha_fill=255, color=(50, 100, 150))
    infos = classify_layers([layer])
    assert infos[0].mean_rgb == (50, 100, 150)


def test_classify_returns_file_index():
    layers = [_make_layer(100, 100) for _ in range(3)]
    infos = classify_layers(layers)
    assert [i.index for i in infos] == [0, 1, 2]


def test_confidence_is_between_zero_and_one():
    layer = _make_layer(640, 480, alpha_fill=255)
    infos = classify_layers([layer])
    assert 0.0 <= infos[0].confidence <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decomposer.classifier'`

- [ ] **Step 3: Write minimal implementation**

```python
# decomposer/classifier.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_classifier.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add decomposer/classifier.py tests/test_classifier.py
git commit -m "feat: add heuristic layer classifier"
```

---

### Task 3: Manifest Schema and Writer

**Files:**
- Create: `tests/test_manifest.py`
- Create: `decomposer/manifest.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest.py
import json
from pathlib import Path

from PIL import Image

from decomposer.classifier import LayerInfo
from decomposer.manifest import Manifest, LayerEntry, write_manifest


def test_manifest_serializes_to_json():
    m = Manifest(
        source="banner.jpg",
        source_dimensions=(1920, 1080),
        resolution=640,
        steps=8,
        layers_requested=3,
        backend="mlx",
        seed=None,
        wall_time_seconds=300.5,
        quality_warnings=[],
        layers=[
            LayerEntry(
                file="layer_0.png",
                index=0,
                classification="background",
                confidence=0.92,
                alpha_coverage=1.0,
                bounding_box=(0, 0, 640, 480),
                mean_rgb=(45, 82, 130),
                file_size_bytes=1148576,
            ),
        ],
    )
    data = json.loads(m.model_dump_json())
    assert data["source"] == "banner.jpg"
    assert data["layers"][0]["classification"] == "background"
    assert data["seed"] is None


def test_write_manifest_creates_file(tmp_path):
    m = Manifest(
        source="test.png",
        source_dimensions=(100, 100),
        resolution=640,
        steps=8,
        layers_requested=1,
        backend="mlx",
        seed=42,
        wall_time_seconds=10.0,
        quality_warnings=[],
        layers=[],
    )
    write_manifest(m, tmp_path / "manifest.json")
    loaded = json.loads((tmp_path / "manifest.json").read_text())
    assert loaded["source"] == "test.png"
    assert loaded["seed"] == 42


def test_layer_entry_from_layer_info():
    info = LayerInfo(
        index=0,
        classification="background",
        confidence=0.95,
        alpha_coverage=1.0,
        bounding_box=(0, 0, 640, 480),
        mean_rgb=(50, 100, 150),
    )
    entry = LayerEntry.from_layer_info(info, file="layer_0.png", file_size_bytes=12345)
    assert entry.file == "layer_0.png"
    assert entry.file_size_bytes == 12345
    assert entry.classification == "background"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decomposer.manifest'`

- [ ] **Step 3: Write minimal implementation**

```python
# decomposer/manifest.py
from __future__ import annotations

import json
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_manifest.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add decomposer/manifest.py tests/test_manifest.py
git commit -m "feat: add manifest schema and writer"
```

---

### Task 4: Post-Decomposition Quality Validation

**Files:**
- Create: `tests/test_quality.py`
- Create: `decomposer/quality.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_quality.py
import numpy as np
import pytest
from PIL import Image

from decomposer.quality import validate_layers, QualityWarning


def _make_layer(w: int, h: int, alpha: int = 255) -> Image.Image:
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[..., :3] = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    arr[..., 3] = alpha
    return Image.fromarray(arr, "RGBA")


def test_all_opaque_layers_warns_non_degeneracy():
    layers = [_make_layer(100, 100, alpha=255) for _ in range(3)]
    warnings = validate_layers(layers)
    assert any(w.code == "all_opaque" for w in warnings)


def test_diverse_layers_no_warning():
    bg = _make_layer(100, 100, alpha=255)
    fg = _make_layer(100, 100, alpha=0)
    arr = np.asarray(fg)
    arr2 = arr.copy()
    arr2[20:80, 20:80, 3] = 255
    arr2[20:80, 20:80, :3] = [255, 0, 0]
    fg2 = Image.fromarray(arr2, "RGBA")
    warnings = validate_layers([bg, fg2])
    assert not any(w.code == "all_opaque" for w in warnings)


def test_all_transparent_layer_warns():
    layers = [_make_layer(100, 100, alpha=0)]
    warnings = validate_layers(layers)
    assert any(w.code == "empty_layer" for w in warnings)


def test_identical_layers_warns_low_diversity():
    layer = _make_layer(100, 100, alpha=255)
    layers = [layer.copy(), layer.copy(), layer.copy()]
    warnings = validate_layers(layers)
    assert any(w.code == "low_diversity" for w in warnings)


def test_warning_has_message():
    layers = [_make_layer(100, 100, alpha=0)]
    warnings = validate_layers(layers)
    assert len(warnings) > 0
    assert len(warnings[0].message) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_quality.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decomposer.quality'`

- [ ] **Step 3: Write minimal implementation**

```python
# decomposer/quality.py
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
            message="All layers are >90% opaque — decomposition may have failed",
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
                    break
            else:
                continue
            break

    return warnings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quality.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add decomposer/quality.py tests/test_quality.py
git commit -m "feat: add post-decomposition quality validation"
```

---

### Task 5: Workflow Orchestrator

**Files:**
- Create: `tests/test_workflow.py`
- Create: `decomposer/workflow.py`

This is the central module that chains all components. Uses `FakeBackend` in tests to avoid loading real models.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_workflow.py
import json
from pathlib import Path

import pytest
from PIL import Image

from decomposer.workflow import run_workflow, WorkflowResult, PreflightError


def test_workflow_produces_layers_and_manifest(tmp_path):
    img = Image.new("RGB", (256, 256), (100, 150, 200))
    src = tmp_path / "input.jpg"
    img.save(src)
    out = tmp_path / "output"

    result = run_workflow(
        image_path=src,
        output_dir=out,
        layers=3,
        resolution=640,
        steps=8,
        seed=42,
        backend_name="fake",
        trace=False,
    )

    assert result.success
    assert len(result.layer_files) == 3
    assert all(f.exists() for f in result.layer_files)
    assert result.manifest_path.exists()

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["source"] == "input.jpg"
    assert manifest["backend"] == "fake"
    assert manifest["seed"] == 42
    assert len(manifest["layers"]) == 3
    assert manifest["layers"][0]["classification"] in [
        "background", "hero_image", "text", "logo",
        "cta_button", "overlay", "unknown",
    ]


def test_workflow_rejects_tiny_image(tmp_path):
    img = Image.new("RGB", (32, 32), (0, 0, 0))
    src = tmp_path / "tiny.png"
    img.save(src)

    result = run_workflow(
        image_path=src,
        output_dir=tmp_path / "out",
        layers=3,
        resolution=640,
        steps=8,
        backend_name="fake",
    )
    assert not result.success
    assert "64" in result.error


def test_workflow_creates_output_dir(tmp_path):
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    src = tmp_path / "img.png"
    img.save(src)
    out = tmp_path / "nested" / "output"

    result = run_workflow(
        image_path=src,
        output_dir=out,
        layers=3,
        resolution=640,
        steps=8,
        backend_name="fake",
    )
    assert result.success
    assert out.exists()


def test_workflow_includes_quality_warnings(tmp_path):
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    src = tmp_path / "img.png"
    img.save(src)

    result = run_workflow(
        image_path=src,
        output_dir=tmp_path / "out",
        layers=3,
        resolution=640,
        steps=8,
        backend_name="fake",
    )
    assert result.success
    manifest = json.loads(result.manifest_path.read_text())
    assert "quality_warnings" in manifest


def test_workflow_with_trace(tmp_path):
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    src = tmp_path / "img.png"
    img.save(src)
    out = tmp_path / "out"

    result = run_workflow(
        image_path=src,
        output_dir=out,
        layers=3,
        resolution=640,
        steps=8,
        backend_name="fake",
        trace=True,
    )
    assert result.success
    assert (out / "trace.json").exists()


def test_workflow_nonexistent_image(tmp_path):
    result = run_workflow(
        image_path=tmp_path / "nope.jpg",
        output_dir=tmp_path / "out",
        layers=3,
        resolution=640,
        steps=8,
        backend_name="fake",
    )
    assert not result.success
    assert "not found" in result.error.lower() or "No such file" in result.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_workflow.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'decomposer.workflow'`

- [ ] **Step 3: Write minimal implementation**

```python
# decomposer/workflow.py
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from decomposer.classifier import classify_layers, LayerInfo
from decomposer.config import get_settings
from decomposer.core.backend import FakeBackend, InferenceBackend
from decomposer.core.perfetto import report_to_json
from decomposer.core.xray import Tracer
from decomposer.manifest import LayerEntry, Manifest, write_manifest
from decomposer.preprocess import PreprocessError, preprocess
from decomposer.quality import validate_layers

logger = logging.getLogger(__name__)


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


def _get_backend(name: str) -> InferenceBackend:
    if name == "fake":
        return FakeBackend(latency_ms=50)
    if name == "mlx":
        from decomposer.core.mlx_backend import MlxBackend
        return MlxBackend(settings=get_settings())
    from decomposer.core.mps_backend import MpsBackend
    return MpsBackend()


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

    output_dir.mkdir(parents=True, exist_ok=True)
    tracer = Tracer(run_id=f"run-{int(time.time())}")
    backend = _get_backend(backend_name)

    try:
        result_layers = backend.decompose(
            image, layers=layers, resolution=resolution,
            steps=steps, seed=seed, tracer=tracer,
        )
    except Exception as e:
        return WorkflowResult(success=False, error=f"Inference failed: {e}")

    layer_files: list[Path] = []
    for i, layer_img in enumerate(result_layers):
        path = output_dir / f"layer_{i}.png"
        layer_img.save(path)
        layer_files.append(path)

    quality_warns = validate_layers(result_layers)
    warning_messages = [w.message for w in quality_warns]

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workflow.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add decomposer/workflow.py tests/test_workflow.py
git commit -m "feat: add workflow orchestrator"
```

---

### Task 6: CLI `run` Command

**Files:**
- Modify: `decomposer/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Add to the existing test file:

```python
# Append to tests/test_cli.py

def test_run_with_fake_produces_manifest(tmp_path):
    from PIL import Image
    src = tmp_path / "photo.jpg"
    Image.new("RGB", (256, 256), (100, 150, 200)).save(src)
    out = tmp_path / "out"
    result = runner.invoke(app, [
        "run", str(src), "--layers", "3", "-o", str(out), "--backend", "fake",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "manifest.json").exists()
    import json
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["source"] == "photo.jpg"
    assert len(manifest["layers"]) == 3


def test_run_rejects_tiny_image(tmp_path):
    from PIL import Image
    src = tmp_path / "tiny.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(src)
    result = runner.invoke(app, [
        "run", str(src), "--backend", "fake",
    ])
    assert result.exit_code == 1


def test_run_batch_multiple_images(tmp_path):
    from PIL import Image
    src1 = tmp_path / "a.jpg"
    src2 = tmp_path / "b.jpg"
    Image.new("RGB", (128, 128), (100, 0, 0)).save(src1)
    Image.new("RGB", (128, 128), (0, 100, 0)).save(src2)
    out = tmp_path / "out"
    result = runner.invoke(app, [
        "run", str(src1), str(src2), "-o", str(out), "--backend", "fake",
    ])
    assert result.exit_code == 0, result.output
    assert (out / "a" / "manifest.json").exists()
    assert (out / "b" / "manifest.json").exists()


def test_run_default_output_dir(tmp_path, monkeypatch):
    from PIL import Image
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "img.png"
    Image.new("RGB", (128, 128), (0, 0, 0)).save(src)
    result = runner.invoke(app, [
        "run", str(src), "--backend", "fake",
    ])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "output" / "img" / "manifest.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py::test_run_with_fake_produces_manifest -v`
Expected: FAIL with `No such command 'run'`

- [ ] **Step 3: Add `run` command to cli.py**

Add the following to `decomposer/cli.py`, after the existing `decompose` command and before `convert_to_mlx`:

```python
@app.command()
def run(
    images: Annotated[list[Path], typer.Argument(exists=True, readable=True)],
    layers: int = 6,
    resolution: int = 640,
    steps: int = 8,
    seed: Optional[int] = None,
    out: Annotated[Path, typer.Option("-o", "--output")] = Path("./output"),
    trace: bool = False,
    backend_name: Annotated[str, typer.Option("--backend", help="Backend: mlx, mps, or fake")] = "mlx",
) -> None:
    """Decompose one or more images into classified RGBA layers with metadata."""
    configure_logging(get_settings())
    from decomposer.workflow import run_workflow

    failed = 0
    for image_path in images:
        if len(images) > 1:
            image_out = out / image_path.stem
        else:
            image_out = out / image_path.stem if out == Path("./output") else out

        console.print(f"[bold]Processing[/bold] {image_path.name} → {image_out}")

        result = run_workflow(
            image_path=image_path,
            output_dir=image_out,
            layers=layers,
            resolution=resolution,
            steps=steps,
            seed=seed,
            backend_name=backend_name,
            trace=trace,
        )

        if not result.success:
            console.print(f"[red]FAILED[/red] {image_path.name}: {result.error}")
            failed += 1
            continue

        console.print(
            f"[green]Done[/green] {image_path.name} — "
            f"{len(result.layer_files)} layers, {result.wall_time_seconds:.1f}s"
        )
        if result.quality_warnings:
            for w in result.quality_warnings:
                console.print(f"  [yellow]Warning:[/yellow] {w}")

    if failed:
        raise typer.Exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py -v`
Expected: All tests PASS (existing + 4 new)

- [ ] **Step 5: Commit**

```bash
git add decomposer/cli.py tests/test_cli.py
git commit -m "feat: add decomposer run command"
```

---

### Task 7: MLX Performance — Apply mx.compile

**Files:**
- Modify: `decomposer/core/mlx_backend.py`
- Modify: `decomposer/mlx_dit/transformer.py`

This task applies `mx.compile` to the MLX transformer's `__call__` method. No unit test needed — this is a performance optimization verified by running the pipeline and comparing trace times.

- [ ] **Step 1: Read current transformer __call__ signature**

Run: `grep -n "def __call__" decomposer/mlx_dit/transformer.py`

Note the method signature — `mx.compile` wraps it to JIT-compile the computation graph.

- [ ] **Step 2: Apply mx.compile in MlxBackend after loading the model**

In `decomposer/core/mlx_backend.py`, inside the `with t.stage("load_dit"):` block, after loading the model, add the compile call:

```python
        with t.stage("load_dit"):
            from decomposer.mlx_dit.loader import load_mlx_transformer
            model = load_mlx_transformer(self.settings.mlx_weights_dir)
            import mlx.core as mx
            model.__call__ = mx.compile(model.__call__)
```

- [ ] **Step 3: Verify the fake backend tests still pass**

Run: `uv run pytest tests/test_workflow.py tests/test_cli.py -v`
Expected: All PASS (mx.compile is only triggered with real MLX backend)

- [ ] **Step 4: Commit**

```bash
git add decomposer/core/mlx_backend.py
git commit -m "perf: apply mx.compile to MLX transformer"
```

---

### Task 8: Integration Smoke Test with FakeBackend

**Files:**
- No new files — runs existing tests end-to-end

This task verifies the full pipeline works end-to-end before shipping.

- [ ] **Step 1: Run all tests**

Run: `uv run pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run the CLI command manually**

Create a test image and run the full workflow:

```bash
uv run python -c "from PIL import Image; Image.new('RGB', (512, 512), (100, 150, 200)).save('/tmp/test_master.jpg')"
uv run decomposer run /tmp/test_master.jpg --backend fake -o /tmp/test_master_out --trace
```

Expected:
- Exit code 0
- `/tmp/test_master_out/test_master/manifest.json` exists with valid JSON
- Layer files exist
- trace.json exists (because `--trace` flag)

- [ ] **Step 3: Verify manifest content**

```bash
cat /tmp/test_master_out/test_master/manifest.json | python -m json.tool
```

Expected: Valid JSON with `source`, `layers` array, `quality_warnings`, `backend: "fake"`, classification labels on each layer.

- [ ] **Step 4: Run batch mode**

```bash
uv run python -c "from PIL import Image; Image.new('RGB', (256, 256), (200, 50, 50)).save('/tmp/test_b.jpg')"
uv run decomposer run /tmp/test_master.jpg /tmp/test_b.jpg --backend fake -o /tmp/batch_out
```

Expected: Two subdirectories in `/tmp/batch_out/`, each with manifest.json and layer PNGs.

- [ ] **Step 5: Lint check**

Run: `uv run ruff check decomposer/ tests/`
Expected: No errors

- [ ] **Step 6: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "fix: integration test fixes"
```

Only run this step if fixes were needed. Skip if everything passed clean.
