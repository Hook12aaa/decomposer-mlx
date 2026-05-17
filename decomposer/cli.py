import json
import logging
import os
import time
from pathlib import Path
from typing import Annotated, Optional

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import typer
from PIL import Image
from rich.console import Console
from rich.table import Table

from decomposer.config import get_settings
from decomposer.core.backend import get_backend
from decomposer.core.perfetto import report_to_json, to_perfetto
from decomposer.core.xray import Tracer
from decomposer.logging_setup import configure_logging
from decomposer.research.cli import research_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(research_app, name="research")
console = Console()
logger = logging.getLogger(__name__)


def _check_hf_auth(settings) -> bool:
    """Return True if HF auth + repo access work; print remediation and return False otherwise."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, GatedRepoError, RepositoryNotFoundError

    token = settings.hf_token.get_secret_value() if settings.hf_token else None
    api = HfApi(token=token)

    repos_to_check = [
        ("model_index", settings.hf_repo),
        ("text_encoder", settings.text_encoder_repo),
        ("gguf", settings.gguf_repo),
    ]
    failed = []
    for label, repo in repos_to_check:
        try:
            api.repo_info(repo)
        except GatedRepoError:
            failed.append((label, repo, "gated, license must be accepted on huggingface.co"))
        except RepositoryNotFoundError:
            failed.append((label, repo, "not found (typo in DECOMPOSER_*_REPO env var?)"))
        except HfHubHTTPError as e:
            failed.append((label, repo, f"HTTP error: {e}"))
        except Exception as e:
            failed.append((label, repo, f"{type(e).__name__}: {e}"))

    if not failed:
        console.print("[green]✓[/green] HF auth OK for all required repos")
        return True

    console.print("[red]✗[/red] HF auth failed for one or more required repos:")
    for label, repo, reason in failed:
        console.print(f"   [yellow]{label}[/yellow] ({repo}): {reason}")
    console.print("")
    console.print("To fix:")
    console.print("  1. Visit each gated repo URL above and click 'Accept license'")
    console.print("  2. Generate a token: https://huggingface.co/settings/tokens (Read scope is enough)")
    console.print("  3. Either: [cyan]huggingface-cli login[/cyan]  OR  set [cyan]DECOMPOSER_HF_TOKEN[/cyan] in your .env")
    return False


@app.command()
def doctor(
    fake: Annotated[bool, typer.Option(help="Use FakeBackend (no model load)")] = False,
) -> None:
    """Smoke-test the install: imports, MPS, Q8 GGUF, mini decomposition."""
    import torch

    configure_logging(get_settings())
    logger.info("doctor command starting fake=%s", fake)
    console.print(f"[bold]decomposer doctor[/bold] fake={fake}")
    console.print(f"  torch: {torch.__version__}")
    console.print(f"  MPS available: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        console.print(f"  MPS recommended max mem: {torch.mps.recommended_max_memory() / 1e9:.1f} GB")

    if not fake:
        if not _check_hf_auth(get_settings()):
            raise typer.Exit(1)

    img_path = get_settings().runs_dir / "doctor_smoke_test.png"
    if not img_path.exists():
        img = Image.new("RGB", (256, 256), (180, 80, 80))
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_path)
    img = Image.open(img_path).convert("RGB")

    backend = get_backend("fake" if fake else "mps")
    t = Tracer(run_id=f"doctor-{int(time.time())}")
    try:
        layers = backend.decompose(img, layers=3, resolution=256, steps=4, tracer=t)
    except Exception as e:
        console.print(f"[red]FAIL[/red]: {e}")
        raise typer.Exit(1)

    _print_report(t.report())
    if len(layers) == 3:
        console.print("[green]PASS[/green] 3 layers returned")
    else:
        console.print(f"[red]FAIL[/red] expected 3 layers, got {len(layers)}")
        raise typer.Exit(1)


@app.command()
def decompose(
    image: Annotated[Path, typer.Argument(exists=True, readable=True)],
    layers: int = 6,
    resolution: int = 640,
    steps: int = 8,
    seed: Optional[int] = None,
    out: Path = Path("./out"),
    trace: bool = True,
    fake: bool = False,
    backend_name: Annotated[str, typer.Option("--backend", help="Backend: mps, mlx, or fake")] = "mps",
) -> None:
    """Decompose IMAGE into N RGBA layers."""
    configure_logging(get_settings())
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(image).convert("RGB")
    effective_backend = "fake" if fake else backend_name
    backend = get_backend(effective_backend)
    t = Tracer(run_id=f"run-{int(time.time())}")

    logger.info(
        "decompose command image=%s out=%s layers=%d resolution=%d steps=%d backend=%s",
        image, out, layers, resolution, steps, effective_backend,
    )
    console.print(f"[bold]Decomposing[/bold] {image} → {out} (layers={layers}, res={resolution}, steps={steps})")
    result = backend.decompose(img, layers=layers, resolution=resolution,
                                steps=steps, seed=seed, tracer=t)

    for i, layer in enumerate(result):
        layer.save(out / f"layer_{i}.png")
    rep = t.report()
    if trace:
        (out / "trace.json").write_text(report_to_json(rep))
        (out / "trace.perfetto.json").write_text(json.dumps(to_perfetto(rep), indent=2))

    _print_report(rep)
    console.print(f"[green]Wrote {len(result)} layers to {out}[/green]")


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
            image_out = out / image_path.stem if out.resolve() == Path("./output").resolve() else out

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
            f"[green]Done[/green] {image_path.name}: "
            f"{len(result.layer_files)} layers, {result.wall_time_seconds:.1f}s"
        )
        if result.quality_warnings:
            for w in result.quality_warnings:
                console.print(f"  [yellow]Warning:[/yellow] {w}")

    if failed:
        raise typer.Exit(1)


@app.command("convert-to-mlx")
def convert_to_mlx(
    gguf: Annotated[Path, typer.Argument(exists=True)],
    output: Path = Path("mlx-weights"),
    bits: int = 8,
) -> None:
    """Convert GGUF weights to MLX quantized safetensors."""
    configure_logging(get_settings())
    from decomposer.mlx_convert.convert import convert_gguf_to_mlx
    console.print(f"[bold]Converting[/bold] {gguf} -> {output} ({bits}-bit)")
    convert_gguf_to_mlx(gguf, output, bits=bits)
    console.print(f"[green]Done. MLX weights at {output}[/green]")


@app.command("diff-traces")
def diff_traces(run_a: Path, run_b: Path) -> None:
    """Compare two trace.json reports side by side."""
    configure_logging(get_settings())
    a = json.loads(Path(run_a).read_text())
    b = json.loads(Path(run_b).read_text())
    by_name_a = {s["name"]: s for s in a["stages"]}
    by_name_b = {s["name"]: s for s in b["stages"]}
    all_names = sorted(set(by_name_a) | set(by_name_b))

    table = Table(title=f"{run_a.name} vs {run_b.name}")
    table.add_column("Stage")
    table.add_column("A wall_ms", justify="right")
    table.add_column("B wall_ms", justify="right")
    table.add_column("Δ wall_ms", justify="right")
    table.add_column("A peak MB", justify="right")
    table.add_column("B peak MB", justify="right")

    for name in all_names:
        sa = by_name_a.get(name, {})
        sb = by_name_b.get(name, {})
        wa = float(sa.get("wall_ms", 0))
        wb = float(sb.get("wall_ms", 0))
        delta = wb - wa
        color = "red" if delta > 0 else "green"
        table.add_row(
            name, f"{wa:.1f}", f"{wb:.1f}",
            f"[{color}]{delta:+.1f}[/{color}]",
            f"{sa.get('mps_alloc_peak_mb', 0):.0f}",
            f"{sb.get('mps_alloc_peak_mb', 0):.0f}",
        )
    console.print(table)


def _print_report(rep) -> None:
    table = Table(title=f"Run {rep.run_id} total {rep.total_wall_ms:.0f} ms")
    table.add_column("Stage")
    table.add_column("Wall ms", justify="right")
    table.add_column("Peak MPS MB", justify="right")
    table.add_column("Fallbacks", justify="right")
    for s in rep.stages:
        table.add_row(s.name, f"{s.wall_ms:.1f}",
                      f"{s.mps_alloc_peak_mb:.0f}", str(s.device_fallbacks))
    console.print(table)
