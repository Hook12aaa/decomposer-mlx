from decomposer.research.ledger import LedgerEntry
from decomposer.research.report import summarize


def _entry(exp_id, decision, delta_pct):
    return LedgerEntry(
        timestamp="2026-05-18T00:00:00Z",
        experiment_id=exp_id,
        baseline_run_id="bl",
        experiment_run_id=f"exp-{exp_id}",
        decision=decision,
        perf={"delta_pct": delta_pct, "baseline_total_wall_ms": 1000.0,
              "experiment_total_wall_ms": 1000.0 * (1 + delta_pct / 100.0)},
        quality={"composite_ssim": 0.95, "per_layer_ssim_matched": 0.90,
                 "non_degenerate": True, "degeneracy_reasons": [], "notes": []},
    )


def test_summarize_lists_merges_and_rejects():
    entries = [
        _entry("a", "MERGE", -50.0),
        _entry("b", "REJECT_QUALITY", -10.0),
        _entry("c", "MERGE", -20.0),
    ]
    summary = summarize(entries)
    assert summary["merged_count"] == 2
    assert summary["rejected_count"] == 1
    merged_ids = [m["experiment_id"] for m in summary["merged"]]
    assert merged_ids == ["a", "c"]


def test_summarize_computes_total_speedup():
    entries = [
        _entry("a", "MERGE", -50.0),
        _entry("b", "MERGE", -30.0),
    ]
    summary = summarize(entries)
    assert summary["total_speedup_pct"] < -50.0
