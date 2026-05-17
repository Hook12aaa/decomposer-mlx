"""Stdlib logging configuration. Idempotent across FastAPI lifespan re-fires and chained CLI commands."""

from __future__ import annotations

import logging
import sys

from decomposer.config import Settings


def configure_logging(settings: Settings | None = None) -> None:
    if settings is None:
        from decomposer.config import get_settings
        settings = get_settings()

    settings.runs_dir.mkdir(parents=True, exist_ok=True)
    log_file = settings.runs_dir / "decomposer.log"

    root = logging.getLogger()
    if root.handlers:
        # FastAPI lifespan can re-fire across reloads; CLI commands can chain.
        # Re-adding handlers would duplicate every log line.
        return

    fmt = "%(asctime)s %(levelname)s %(name)s :: %(message)s"
    formatter = logging.Formatter(fmt)

    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    root.setLevel(logging.INFO)
    root.addHandler(stderr)
    root.addHandler(file_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
