import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Iterator

from decomposer.core.probes import FallbackCounter, mps_alloc_mb, rss_mb
from decomposer.core.types import Report, StageRecord

Listener = Callable[[str, dict[str, Any]], None]


class Tracer:
    """Per-run tracer.

    Fallback-warning capture is scoped to the root ``stage()`` context via
    ``warnings.catch_warnings(record=True)``, so there is no global mutation of
    ``warnings.showwarning``. Note: ``warnings.catch_warnings`` itself is not
    thread-safe per the stdlib docs (it mutates module-level state). In this
    application concurrent inference is serialized by an ``asyncio.Lock`` in
    ``web/app.py``, so concurrent root stages do not occur in practice.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._stages: list[StageRecord] = []
        self._stack: list[StageRecord] = []
        self._listeners: list[Listener] = []

    def subscribe(self, fn: Listener) -> None:
        self._listeners.append(fn)

    def _emit(self, event: str, **payload: Any) -> None:
        for fn in self._listeners:
            fn(event, payload)

    def _attribute_warnings_to_root(self, caught: list, root: StageRecord) -> None:
        if not caught:
            return
        for w in list(caught):
            text = str(w.message)
            if "fell back to CPU" in text or ("MPS:" in text and "fallback" in text):
                root.device_fallbacks += 1

    @contextmanager
    def stage(self, name: str, **extras: Any) -> Iterator[None]:
        is_root = not self._stack
        rec = self._begin(name, extras)
        if is_root:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    yield
                finally:
                    self._attribute_warnings_to_root(caught, rec)
                    self._end(rec)
        else:
            try:
                yield
            finally:
                self._end(rec)

    @contextmanager
    def step(self, name: str, **extras: Any) -> Iterator[None]:
        if not self._stack:
            raise RuntimeError("step() requires an active stage")
        rec = self._begin(name, extras)
        try:
            yield
        finally:
            self._end(rec)

    def annotate(self, **extras: Any) -> None:
        if not self._stack:
            raise RuntimeError("annotate() requires an active stage")
        self._stack[-1].extras.update(extras)

    def _begin(self, name: str, extras: dict[str, Any]) -> StageRecord:
        rec = StageRecord(
            name=name, wall_ms=0.0, gpu_ms=0.0,
            rss_peak_mb=rss_mb(), mps_alloc_peak_mb=mps_alloc_mb(),
            mps_alloc_delta_mb=0.0, device_fallbacks=0, extras=dict(extras),
        )
        self._stack.append(rec)
        self._emit("stage_started", name=name, extras=extras)
        rec._start_ts = time.perf_counter()
        rec._start_mps = mps_alloc_mb()
        return rec

    def _end(self, rec: StageRecord) -> None:
        end_ts = time.perf_counter()
        end_mps = mps_alloc_mb()
        rec.wall_ms = (end_ts - rec._start_ts) * 1000.0
        rec.mps_alloc_peak_mb = max(rec.mps_alloc_peak_mb, end_mps)
        rec.mps_alloc_delta_mb = end_mps - rec._start_mps
        rec.rss_peak_mb = max(rec.rss_peak_mb, rss_mb())
        self._stack.pop()
        if self._stack:
            self._stack[-1].steps.append(rec)
        else:
            self._stages.append(rec)
        self._emit("stage_ended", name=rec.name, wall_ms=rec.wall_ms)

    def report(self) -> Report:
        total = sum(s.wall_ms for s in self._stages)
        return Report(stages=list(self._stages), total_wall_ms=total, run_id=self.run_id)
