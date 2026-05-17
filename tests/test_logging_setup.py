import logging

from decomposer.logging_setup import configure_logging


def _reset_root() -> None:
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)


def test_configure_logging_creates_log_file(tmp_path, monkeypatch):
    monkeypatch.setenv("DECOMPOSER_RUNS_DIR", str(tmp_path))
    _reset_root()

    configure_logging()
    log_file = tmp_path / "decomposer.log"
    assert log_file.exists()

    logger = logging.getLogger("test")
    logger.info("hello from test")

    for h in logging.getLogger().handlers:
        h.flush()

    contents = log_file.read_text()
    assert "hello from test" in contents


def test_configure_logging_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("DECOMPOSER_RUNS_DIR", str(tmp_path))
    _reset_root()

    configure_logging()
    handler_count = len(logging.getLogger().handlers)
    configure_logging()
    assert len(logging.getLogger().handlers) == handler_count
