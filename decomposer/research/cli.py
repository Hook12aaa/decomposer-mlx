from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from decomposer.config import get_settings
from decomposer.research.baseline import load_baseline
from decomposer.research.experiments import load_queue
from decomposer.research.ledger import read_all
from decomposer.research.report import summarize
from decomposer.research.run import StopCondition, run_queue

research_app = typer.Typer(no_args_is_help=True, help="Auto-researcher subcommands")
console = Console()


def _parse_duration(text: str) -> float:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", text.strip().lower())
    if not match:
        raise typer.BadParameter(f"duration must look like '30s', '5m', '8h': got {text!r}")
    value = float(match.group(1))
    return value * {"s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]


def _repo_root() -> Path:
    return Path.cwd()


def _load_baseline_or_die(settings):
    bl_dir = settings.runs_dir / "baseline-latest"
    if not bl_dir.exists():
        console.print(f"[red]No baseline at {bl_dir}. Run `decomposer research baseline` first.[/red]")
        raise typer.Exit(1)
    return load_baseline(bl_dir)


@research_app.command()
def baseline(
    image: Annotated[Path, typer.Option(exists=True, readable=True)] = Path("test_image.jpg"),
    layers: int = 3,
    resolution: int = 640,
    steps: int = 8,
) -> None:
    """Capture a reference run as the no-regression target."""
    settings = get_settings()
    out_dir = settings.runs_dir / "baseline-latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[bold]baseline[/bold] image={image} layers={layers} res={resolution} steps={steps}")
    console.print(f"[yellow]Run `decomposer decompose {image} --layers {layers} --resolution {resolution} --steps {steps} --out {out_dir} --trace` to populate it[/yellow]")


@research_app.command()
def run(
    queue: Annotated[Path, typer.Option(exists=True, readable=True)],
    budget: str = "8h",
    target_latency: str = "30s",
    max_experiments: int = 10,
) -> None:
    """Execute experiments from the queue (sequential)."""
    settings = get_settings()
    baseline = _load_baseline_or_die(settings)
    hypotheses = load_queue(queue)
    stop = StopCondition(
        budget_seconds=_parse_duration(budget),
        target_latency_seconds=_parse_duration(target_latency),
        max_experiments=max_experiments,
    )
    ledger_path = settings.runs_dir / "ledger.jsonl"
    entries = run_queue(
        hypotheses, baseline, _repo_root(),
        ledger_path=ledger_path,
        image_path=str(Path("test_image.jpg").resolve()),
        stop=stop,
    )
    console.print(f"[bold]run complete[/bold] {len(entries)} experiments processed")
    merged = sum(1 for e in entries if e.decision == "MERGE")
    console.print(f"  {merged} merged, {len(entries) - merged} not merged")


@research_app.command()
def report() -> None:
    """Human-readable summary of runs/ledger.jsonl."""
    settings = get_settings()
    ledger_path = settings.runs_dir / "ledger.jsonl"
    entries = read_all(ledger_path)
    if not entries:
        console.print("No ledger entries.")
        return
    summary = summarize(entries)
    table = Table(title="Experiment summary")
    table.add_column("metric")
    table.add_column("value")
    table.add_row("total_count", str(summary["total_count"]))
    table.add_row("merged_count", str(summary["merged_count"]))
    table.add_row("rejected_count", str(summary["rejected_count"]))
    table.add_row("review_count", str(summary["review_count"]))
    table.add_row("total_speedup_pct", f"{summary['total_speedup_pct']:.1f}%")
    console.print(table)


@research_app.command()
def replay(experiment_id: str) -> None:
    """Re-run a specific experiment from the ledger."""
    settings = get_settings()
    ledger_path = settings.runs_dir / "ledger.jsonl"
    entries = read_all(ledger_path)
    matching = [e for e in entries if e.experiment_id == experiment_id]
    if not matching:
        console.print(f"[red]no experiment {experiment_id!r} in ledger[/red]")
        raise typer.Exit(1)
    entry = matching[-1]
    console.print(f"Most recent run for {experiment_id!r}:")
    console.print(f"  decision: {entry.decision}")
    console.print(f"  perf delta_pct: {entry.perf.get('delta_pct'):.2f}%")
    console.print(f"  quality: composite={entry.quality.get('composite_ssim'):.3f} "
                  f"per_layer={entry.quality.get('per_layer_ssim_matched'):.3f}")
    console.print("[yellow]Re-execution is not yet wired; this prints the historical result.[/yellow]")
