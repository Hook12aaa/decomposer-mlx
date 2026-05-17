from typer.testing import CliRunner
from decomposer.cli import app

runner = CliRunner()


def test_research_help_lists_subcommands():
    result = runner.invoke(app, ["research", "--help"])
    assert result.exit_code == 0
    for sub in ("baseline", "run", "report", "replay"):
        assert sub in result.output


def test_research_report_on_empty_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("DECOMPOSER_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["research", "report"])
    assert result.exit_code == 0
    assert "No ledger entries" in result.output or "merged_count" in result.output


def test_research_run_loads_queue_and_invokes_loop(tmp_path, monkeypatch):
    queue_file = tmp_path / "q.yaml"
    queue_file.write_text(
        '''
experiments:
  - id: x
    description: ""
    apply:
      kind: env_var
      params: {}
    quality_bounds:
      composite_ssim_min: 0.5
      per_layer_ssim_min: 0.5
'''
    )
    called = {}

    def fake_run_queue(*args, **kwargs):
        called["yes"] = True
        called["stop"] = kwargs.get("stop")
        return []

    monkeypatch.setenv("DECOMPOSER_RUNS_DIR", str(tmp_path))
    monkeypatch.setattr("decomposer.research.cli.run_queue", fake_run_queue)
    monkeypatch.setattr(
        "decomposer.research.cli._load_baseline_or_die",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr("decomposer.research.cli._repo_root", lambda: tmp_path)

    result = runner.invoke(
        app,
        [
            "research",
            "run",
            "--queue",
            str(queue_file),
            "--budget",
            "1h",
            "--target-latency",
            "30s",
            "--max-experiments",
            "5",
        ],
    )
    assert called.get("yes") is True
    assert called["stop"].max_experiments == 5
    assert result.exit_code == 0
