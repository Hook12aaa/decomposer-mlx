from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageRecord:
    name: str
    wall_ms: float
    gpu_ms: float
    rss_peak_mb: float
    mps_alloc_peak_mb: float
    mps_alloc_delta_mb: float
    device_fallbacks: int
    extras: dict[str, Any] = field(default_factory=dict)
    steps: list["StageRecord"] = field(default_factory=list)
    _start_ts: float = 0.0
    _start_mps: float = 0.0


@dataclass
class Report:
    stages: list[StageRecord]
    total_wall_ms: float
    run_id: str

    def total_fallbacks(self) -> int:
        return sum(s.device_fallbacks for s in self.stages)

    def peak_mps_alloc_mb(self) -> float:
        return max((s.mps_alloc_peak_mb for s in self.stages), default=0.0)
