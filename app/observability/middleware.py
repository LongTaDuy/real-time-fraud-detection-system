"""HTTP middleware: request ID, timing, access + unexpected error logs."""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

log = structlog.get_logger("app.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign ``X-Request-ID``, bind structlog context, log each request and unexpected errors."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)
        request.state.request_id = request_id

        start = time.perf_counter()

        try:
            response = await call_next(request)
        except StarletteHTTPException as exc:
            duration_ms = max(0, int((time.perf_counter() - start) * 1000))
            log.info(
                "request_completed",
                method=request.method,
                path=request.url.path,
                status_code=exc.status_code,
                duration_ms=duration_ms,
            )
            structlog.contextvars.clear_contextvars()
            raise
        except Exception:
            # Unhandled errors: ``register_safe_error_handlers`` logs once with request_id.
            structlog.contextvars.clear_contextvars()
            raise

        duration_ms = max(0, int((time.perf_counter() - start) * 1000))
        response.headers["X-Request-ID"] = request_id
        log.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        structlog.contextvars.clear_contextvars()
        return response
