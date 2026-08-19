"""Process-wide console/JSON logging with request correlation."""

from __future__ import annotations

import json
import logging
import logging.config
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

_request_id: ContextVar[str] = ContextVar("request_id", default="-")


def bind_request_id(request_id: str) -> None:
    """Bind a request identifier to the current async context."""
    _request_id.set(request_id)


def current_request_id() -> str:
    """Return the current request identifier or a stable missing marker."""
    return _request_id.get()


class RequestContextFilter(logging.Filter):
    """Attach context-variable fields to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Inject the current request ID and keep the record."""
        record.request_id = current_request_id()
        return True


class JSONFormatter(logging.Formatter):
    """Render bounded structured logs without an additional dependency."""

    _context_fields = (
        "request_id",
        "actor",
        "action",
        "run_id",
        "reason",
        "dimension",
        "principal_kind",
    )

    def format(self, record: logging.LogRecord) -> str:
        """Serialize one log record as a JSON object."""
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in self._context_fields:
            value = getattr(record, field, None)
            if value not in {None, "", "-"}:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Configure root and uvicorn loggers from LOG_FORMAT."""
    log_format = os.environ.get("LOG_FORMAT", "console").strip().lower()
    if log_format not in {"console", "json"}:
        log_format = "console"
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    formatter = "json" if log_format == "json" else "console"
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {"request_context": {"()": RequestContextFilter}},
            "formatters": {
                "console": {
                    "format": "%(asctime)s %(levelname)s %(name)s [request_id=%(request_id)s] %(message)s"
                },
                "json": {"()": JSONFormatter},
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": formatter,
                    "filters": ["request_context"],
                    "stream": "ext://sys.stderr",
                }
            },
            "root": {"handlers": ["default"], "level": level},
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": level, "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": level, "propagate": False},
            },
        }
    )


__all__ = [
    "JSONFormatter",
    "RequestContextFilter",
    "bind_request_id",
    "configure_logging",
    "current_request_id",
]
