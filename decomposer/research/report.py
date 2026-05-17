from __future__ import annotations

from dataclasses import asdict
from typing import Any

from decomposer.research.ledger import LedgerEntry


def summarize(entries: list[LedgerEntry]) -> dict[str, Any]:
    merged = [e for e in entries if e.decision == "MERGE"]
    rejected = [e for e in entries if e.decision.startswith("REJECT_")]
    review = [e for e in entries if e.decision == "KEEP_FOR_REVIEW"]

    cumulative_factor = 1.0
    for m in merged:
        delta_pct = float(m.perf.get("delta_pct", 0.0))
        cumulative_factor *= (1 + delta_pct / 100.0)
    total_speedup_pct = (cumulative_factor - 1) * 100.0

    return {
        "total_count": len(entries),
        "merged_count": len(merged),
        "rejected_count": len(rejected),
        "review_count": len(review),
        "merged": [asdict(m) for m in merged],
        "rejected": [
            {"experiment_id": e.experiment_id, "decision": e.decision,
             "rejection_reason": e.rejection_reason}
            for e in rejected
        ],
        "review": [{"experiment_id": e.experiment_id} for e in review],
        "total_speedup_pct": total_speedup_pct,
    }
