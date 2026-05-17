from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class HypothesisKind(str, Enum):
    LORA_LOAD = "lora_load"
    CODE_PATCH = "code_patch"
    ENV_VAR = "env_var"
    SETTING_CHANGE = "setting_change"
    SCHEDULER_SWAP = "scheduler_swap"


@dataclass
class Apply:
    kind: HypothesisKind
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    id: str
    description: str
    apply: Apply
    overrides: dict[str, Any] = field(default_factory=dict)
    predicted_delta: dict[str, str] = field(default_factory=dict)
    quality_bounds: dict[str, float] = field(default_factory=dict)


def load_queue(path: Path) -> list[Hypothesis]:
    data = yaml.safe_load(path.read_text())
    experiments = data.get("experiments", [])
    queue: list[Hypothesis] = []
    for i, entry in enumerate(experiments):
        if "id" not in entry:
            raise ValueError(f"experiment {i} missing 'id' field")
        apply_block = entry.get("apply") or {}
        kind_str = apply_block.get("kind")
        if kind_str not in {k.value for k in HypothesisKind}:
            raise ValueError(
                f"experiment {entry['id']!r}: unknown hypothesis kind {kind_str!r}; "
                f"valid: {[k.value for k in HypothesisKind]}"
            )
        params = {k: v for k, v in apply_block.items() if k != "kind"}
        h = Hypothesis(
            id=entry["id"],
            description=entry.get("description", ""),
            apply=Apply(kind=HypothesisKind(kind_str), params=params),
            overrides=entry.get("overrides") or {},
            predicted_delta=entry.get("predicted_delta") or {},
            quality_bounds=entry.get("quality_bounds") or {},
        )
        queue.append(h)
    return queue
