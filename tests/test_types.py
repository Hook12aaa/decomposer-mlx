from decomposer.core.types import Report, StageRecord


def test_stage_record_required_fields():
    r = StageRecord(name="load_dit", wall_ms=42.0, gpu_ms=40.0,
                    rss_peak_mb=1024.0, mps_alloc_peak_mb=512.0,
                    mps_alloc_delta_mb=500.0, device_fallbacks=0,
                    extras={"quant": "q8_gguf"})
    assert r.name == "load_dit"
    assert r.extras["quant"] == "q8_gguf"


def test_report_aggregates_stages():
    r1 = StageRecord(name="a", wall_ms=10.0, gpu_ms=5.0, rss_peak_mb=100.0,
                     mps_alloc_peak_mb=50.0, mps_alloc_delta_mb=0.0,
                     device_fallbacks=0, extras={})
    r2 = StageRecord(name="b", wall_ms=20.0, gpu_ms=15.0, rss_peak_mb=120.0,
                     mps_alloc_peak_mb=80.0, mps_alloc_delta_mb=30.0,
                     device_fallbacks=2, extras={})
    rep = Report(stages=[r1, r2], total_wall_ms=30.0, run_id="run-1")
    assert rep.total_fallbacks() == 2
    assert rep.peak_mps_alloc_mb() == 80.0
