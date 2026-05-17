from pathlib import Path

from decomposer.research.ledger import LedgerEntry, append, read_all


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "ledger.jsonl"
    entry = LedgerEntry(
        timestamp="2026-05-18T15:42:00Z",
        experiment_id="lightning-4step",
        baseline_run_id="bl-1",
        experiment_run_id="exp-1",
        decision="MERGE",
        perf={"delta_pct": -50.3},
        quality={"composite_ssim": 0.94, "per_layer_ssim_matched": 0.88,
                 "non_degenerate": True, "degeneracy_reasons": [], "notes": []},
        merged_commit_sha="abc",
        hypothesis_summary="Lightning LoRA 4 steps",
    )
    append(path, entry)
    append(path, entry)
    entries = read_all(path)
    assert len(entries) == 2
    assert entries[0].experiment_id == "lightning-4step"


def test_read_all_empty_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text("")
    assert read_all(path) == []


def test_read_all_missing_file(tmp_path):
    path = tmp_path / "ledger.jsonl"
    assert read_all(path) == []
