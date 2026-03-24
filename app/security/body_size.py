"""Reject oversized request bodies using ``Content-Length`` (cheap pre-read check)."""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Return 413 when ``Content-Length`` exceeds the configured maximum.

    Requests without ``Content-Length`` (e.g. some chunked uploads) are not pre-rejected;
    keep payloads small and use HTTPS in production.
    """

    def __init__(self, app, *, max_content_length_bytes: int) -> None:
        super().__init__(app)
        self._max = max_content_length_bytes

    async def dispatch(self, request: Request, call_next) -> Response:
        if self._max <= 0:
            return await call_next(request)
        if request.method not in ("POST", "PUT", "PATCH"):
            return await call_next(request)
        raw = request.headers.get("content-length")
        if raw is None:
            return await call_next(request)
        try:
            length = int(raw)
        except ValueError:
            return JSONResponse(
                status_code=400,
                content={"detail": "Invalid Content-Length header"},
            )
        if length > self._max:
            rid = getattr(request.state, "request_id", None)
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Request body too large",
                    "request_id": rid,
                    "max_bytes": self._max,
                },
            )
        return await call_next(request)
