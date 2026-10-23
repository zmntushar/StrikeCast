"""Structured (key=value) logging."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


class KVFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S')} level={record.levelname} logger={record.name} msg=\"{record.getMessage()}\""
        extra = getattr(record, "kv", None)
        if extra:
            base += " " + " ".join(f"{k}={v}" for k, v in extra.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        from app.core.config import get_settings

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(KVFormatter())
        root = logging.getLogger("strikecast")
        root.addHandler(handler)
        root.setLevel(get_settings().log_level.upper())
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(f"strikecast.{name}")


def log_kv(logger: logging.Logger, level: int, msg: str, **kv: object) -> None:
    logger.log(level, msg, extra={"kv": kv})
