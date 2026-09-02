"""Logging setup.

Two sinks: a human-readable console stream and a machine-readable JSON Lines
file under ``logs/``. Forensic tooling should leave an auditable trail, so the
JSON sink records timestamp, level, logger, message, and any structured extras
attached via ``logger.info(msg, extra={"extra_fields": {...}})``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

_CONFIGURED = False
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extras = getattr(record, "extra_fields", None)
        if isinstance(extras, dict):
            payload.update(extras)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and key != "extra_fields":
                payload[key] = _jsonable(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging(
    level: str = "INFO",
    log_dir: str | Path | None = None,
    *,
    run_name: str = "neurofence",
    json_file: bool = True,
    console: bool = True,
    force: bool = False,
) -> Path | None:
    """Configure root logging once. Returns the JSON log path, if written."""
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED and not force:
        return None
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(ConsoleFormatter())
        root.addHandler(stream)

    log_path: Path | None = None
    if json_file and log_dir is not None:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"{run_name}.jsonl"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonLineFormatter())
        root.addHandler(file_handler)

    # Transformers is chatty at INFO and drowns out our own output.
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _CONFIGURED = True
    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def _jsonable(value: object) -> object:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)
