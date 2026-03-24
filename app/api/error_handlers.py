"""Last-resort handler so 500 responses never echo raw exception text to clients."""

from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

log = structlog.get_logger(__name__)


def register_safe_error_handlers(app: FastAPI) -> None:
    """Register a generic ``Exception`` handler (more specific FastAPI handlers still win via MRO)."""

    @app.exception_handler(Exception)
    async def safe_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = getattr(request.state, "request_id", None)
        log.exception(
            "internal_server_error",
            request_id=rid,
            exc_type=type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An unexpected error occurred.",
                "request_id": rid,
            },
        )
