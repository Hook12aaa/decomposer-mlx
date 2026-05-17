from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LedgerEntry:
    timestamp: str
    experiment_id: str
    baseline_run_id: str
    experiment_run_id: str
    decision: str
    perf: dict[str, Any]
    quality: dict[str, Any]
    hypothesis_summary: str = ""
    merged_commit_sha: str | None = None
    worktree_path: str | None = None
    human_audit_pending: bool = False
    rejection_reason: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def append(path: Path, entry: LedgerEntry) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(entry)))
        f.write("\n")


def read_all(path: Path) -> list[LedgerEntry]:
    if not path.exists():
        return []
    entries: list[LedgerEntry] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        entries.append(LedgerEntry(**data))
    return entries
