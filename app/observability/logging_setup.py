"""Structured logging via structlog + stdlib (readable in local terminals, easy to parse later)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config import Settings

_configured = False


def get_logger(name: str | None = None):
    """Return a structlog logger (bind key=value pairs for context)."""
    return structlog.get_logger(name)


def configure_observability(settings: Settings) -> None:
    """Idempotent: wire structlog to stdlib, tune third-party noise, set level from settings."""
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Single stream; message body comes from structlog processors (no duplicate prefixes).
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
        force=True,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
    ]

    structlog.configure(
        processors=shared_processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Align stdlib loggers with app level; reduce duplicate HTTP access lines (we log in middleware).
    for name in ("uvicorn", "uvicorn.error"):
        logging.getLogger(name).setLevel(level)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    _configured = True
