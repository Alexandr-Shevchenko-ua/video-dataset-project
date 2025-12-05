# STATUS: not executed, pseudocode – requires adaptation
"""Logging helpers shared across modules."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logging(output_dir: Path, level: str = "INFO") -> Path:
    """Configure logging to file + console.

    Args:
        output_dir: Directory where logs should be stored.
        level: Log level name.

    Returns:
        Path to the log file.
    """

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "pipeline.log"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug("Logging initialized at %s", log_path)
    return log_path


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper for module loggers."""

    return logging.getLogger(name)


class GradioLogBridge(logging.Handler):
    """Logging handler that forwards records into a Gradio component callback."""

    def __init__(self, callback: Optional[callable]):
        super().__init__()
        self._callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        """Send log line to callback if available."""

        if not self._callback:
            return
        message = self.format(record)
        try:
            self._callback(message)
        except Exception:  # pragma: no cover - defensive
            logging.getLogger(__name__).exception("Failed to emit log to Gradio")
