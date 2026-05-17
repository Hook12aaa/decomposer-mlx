import json

from decomposer.core.perfetto import report_to_json, to_perfetto
from decomposer.core.xray import Tracer


def make_report():
    t = Tracer(run_id="run-test")
    with t.stage("load_dit"):
        with t.stage("inner"):
            pass
    with t.stage("denoise_loop", steps=2):
        with t.step("denoise_step", i=0):
            pass
        with t.step("denoise_step", i=1):
            pass
    return t.report()


def test_report_to_json_serializes_round_trip():
    rep = make_report()
    raw = report_to_json(rep)
    parsed = json.loads(raw)
    assert parsed["run_id"] == "run-test"
    assert len(parsed["stages"]) == 2
    assert parsed["stages"][1]["name"] == "denoise_loop"
    assert len(parsed["stages"][1]["steps"]) == 2


def test_perfetto_emits_trace_events_array():
    rep = make_report()
    trace = to_perfetto(rep)
    assert "traceEvents" in trace
    events = trace["traceEvents"]
    names = [e["name"] for e in events if e["ph"] == "X"]
    assert "load_dit" in names
    assert "denoise_step" in names
    for e in events:
        if e["ph"] == "X":
            assert "ts" in e and "dur" in e and "pid" in e


def test_perfetto_stage_events_are_monotonically_ordered():
    rep = make_report()
    trace = to_perfetto(rep)
    stage_events = [e for e in trace["traceEvents"] if e["cat"] == "stage"]
    for prev, nxt in zip(stage_events, stage_events[1:]):
        assert nxt["ts"] >= prev["ts"] + prev["dur"], (
            f"stage {nxt['name']!r} starts before {prev['name']!r} ends"
        )


def test_perfetto_child_events_fit_inside_parent_window():
    rep = make_report()
    trace = to_perfetto(rep)
    stage_events = [e for e in trace["traceEvents"] if e["cat"] == "stage"]
    step_events = [e for e in trace["traceEvents"] if e["cat"] == "step"]
    for step in step_events:
        parents = [s for s in stage_events
                    if s["ts"] <= step["ts"] < s["ts"] + s["dur"]]
        assert parents, f"step {step['name']!r} has no parent stage window"
        parent = parents[-1]
        assert step["ts"] + step["dur"] <= parent["ts"] + parent["dur"]


def test_report_to_json_strips_private_underscore_fields():
    rep = make_report()
    parsed = json.loads(report_to_json(rep))
    for stage in parsed["stages"]:
        assert "_start_ts" not in stage
        assert "_start_mps" not in stage
        for step in stage.get("steps", []):
            assert "_start_ts" not in step
            assert "_start_mps" not in step
