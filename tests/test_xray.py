import time
import warnings as _warnings_mod

import pytest

from decomposer.core.xray import Tracer


def test_tracer_records_a_stage():
    t = Tracer(run_id="r1")
    with t.stage("load_dit", quant="q8"):
        time.sleep(0.01)
    report = t.report()
    assert len(report.stages) == 1
    s = report.stages[0]
    assert s.name == "load_dit"
    assert s.wall_ms >= 10.0
    assert s.extras["quant"] == "q8"


def test_tracer_records_steps_within_a_stage():
    t = Tracer(run_id="r2")
    with t.stage("denoise_loop", steps=3):
        for i in range(3):
            with t.step("denoise_step", i=i):
                time.sleep(0.005)
    report = t.report()
    assert len(report.stages) == 1
    assert len(report.stages[0].steps) == 3
    assert all(s.name == "denoise_step" for s in report.stages[0].steps)


def test_tracer_annotate_adds_to_current_stage():
    t = Tracer(run_id="r3")
    with t.stage("encode_prompt"):
        t.annotate(token_count=128)
    report = t.report()
    assert report.stages[0].extras["token_count"] == 128


def test_tracer_increments_fallback_count_on_warning():
    import warnings
    t = Tracer(run_id="r-fb")
    with t.stage("denoise_step"):
        warnings.warn("aten::some_op fell back to CPU on MPS", UserWarning)
    report = t.report()
    assert report.stages[0].device_fallbacks == 1


def test_tracer_total_wall_sums_stage_walls():
    t = Tracer(run_id="r4")
    with t.stage("a"):
        time.sleep(0.01)
    with t.stage("b"):
        time.sleep(0.01)
    report = t.report()
    assert report.total_wall_ms >= 20.0


def test_tracer_does_not_mutate_global_showwarning():
    before = _warnings_mod.showwarning
    t = Tracer(run_id="r-noglobal")
    with t.stage("only"):
        pass
    assert _warnings_mod.showwarning is before


def test_tracer_does_not_count_warnings_issued_outside_a_stage():
    import warnings
    t = Tracer(run_id="r-outside")
    warnings.warn("aten::out fell back to CPU on MPS", UserWarning)
    with t.stage("s"):
        warnings.warn("aten::inside fell back to CPU on MPS", UserWarning)
    rep = t.report()
    assert rep.stages[0].device_fallbacks == 1


def test_tracer_records_stage_even_when_body_raises():
    t = Tracer(run_id="r-raise")
    with pytest.raises(ValueError):
        with t.stage("crashy"):
            raise ValueError("boom")
    rep = t.report()
    assert len(rep.stages) == 1
    assert rep.stages[0].name == "crashy"
    import warnings as w
    assert w.showwarning is not None


def test_slow_listener_does_not_inflate_wall_ms():
    t = Tracer(run_id="r-overhead")

    def slow_listener(event, payload):
        time.sleep(0.05)

    t.subscribe(slow_listener)
    with t.stage("instant"):
        pass

    assert t.report().stages[0].wall_ms < 30.0, (
        f"listener overhead leaked into wall_ms: {t.report().stages[0].wall_ms}"
    )


def test_tracer_supports_nested_stages():
    t = Tracer(run_id="r-nest")
    with t.stage("outer"):
        with t.stage("inner"):
            pass
    rep = t.report()
    assert len(rep.stages) == 1
    assert rep.stages[0].name == "outer"
    assert len(rep.stages[0].steps) == 1
    assert rep.stages[0].steps[0].name == "inner"
