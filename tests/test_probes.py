from decomposer.core.probes import FallbackCounter, mps_alloc_mb, rss_mb


def test_rss_mb_returns_positive_number():
    assert rss_mb() > 0


def test_mps_alloc_mb_returns_number():
    assert mps_alloc_mb() >= 0


def test_fallback_counter_starts_at_zero():
    c = FallbackCounter()
    assert c.count == 0


def test_fallback_counter_increments_on_warning():
    c = FallbackCounter()
    c.note("aten::some_op fell back to CPU")
    c.note("aten::another fell back to CPU")
    assert c.count == 2


def test_fallback_counter_ignores_unrelated_warnings():
    c = FallbackCounter()
    c.note("UserWarning: something else")
    assert c.count == 0
