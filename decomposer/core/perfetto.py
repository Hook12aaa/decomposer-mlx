import json
from dataclasses import asdict

from decomposer.core.types import Report, StageRecord


def _clean(obj):
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_clean(x) for x in obj]
    return obj


def report_to_json(report: Report) -> str:
    return json.dumps(_clean(asdict(report)), indent=2, default=str)


def to_perfetto(report: Report) -> dict:
    events: list[dict] = []

    def emit(rec: StageRecord, depth: int, start_us: int) -> int:
        dur_us = int(rec.wall_ms * 1000)
        events.append({
            "name": rec.name,
            "cat": "stage" if depth == 0 else "step",
            "ph": "X",
            "ts": start_us,
            "dur": dur_us,
            "pid": 1,
            "tid": depth,
            "args": {
                "gpu_ms": rec.gpu_ms,
                "mps_alloc_peak_mb": rec.mps_alloc_peak_mb,
                "rss_peak_mb": rec.rss_peak_mb,
                "device_fallbacks": rec.device_fallbacks,
                **rec.extras,
            },
        })
        child_cursor = start_us
        for child in rec.steps:
            child_cursor = emit(child, depth + 1, child_cursor)
        return start_us + dur_us

    cursor = 0
    for s in report.stages:
        cursor = emit(s, 0, cursor)

    return {"traceEvents": events, "displayTimeUnit": "ms"}
