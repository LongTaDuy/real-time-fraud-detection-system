"""Simple sliding-window rate limits per client IP (in-process; document for single-worker MVP)."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import Settings
from app.security.client_ip import get_client_ip

log = structlog.get_logger(__name__)

_FRAUD_POST_PATHS = frozenset(
    {
        "/api/v1/fraud/score",
        "/api/v1/fraud/score-async",
    }
)


class SlidingWindowRateLimitMiddleware(BaseHTTPMiddleware):
    """429 when an IP exceeds per-minute counts (fraud POST bucket + general API bucket)."""

    def __init__(
        self,
        app,
        *,
        settings: Settings,
    ) -> None:
        super().__init__(app)
        self._settings = settings
        self._general: dict[str, deque[float]] = defaultdict(deque)
        self._fraud: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    def _prune(self, dq: deque[float], window_sec: int, now: float) -> None:
        cutoff = now - window_sec
        while dq and dq[0] < cutoff:
            dq.popleft()

    async def dispatch(self, request: Request, call_next) -> Response:
        path = request.url.path
        if not path.startswith("/api/"):
            return await call_next(request)

        s = self._settings
        ip = get_client_ip(request, trust_proxy_headers=s.trust_proxy_headers)
        now = time.monotonic()
        is_fraud_post = request.method == "POST" and path in _FRAUD_POST_PATHS

        async with self._lock:
            if is_fraud_post and s.fraud_post_rate_limit_per_minute > 0:
                dq_f = self._fraud[ip]
                self._prune(dq_f, 60, now)
                if len(dq_f) >= s.fraud_post_rate_limit_per_minute:
                    log.warning("rate_limit_exceeded", tier="fraud_post", client_ip=ip, path=path)
                    rid = getattr(request.state, "request_id", None)
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Too many fraud scoring requests. Try again later.",
                            "request_id": rid,
                        },
                        headers={"Retry-After": "60"},
                    )
                dq_f.append(now)

            if s.api_rate_limit_per_minute > 0:
                dq_g = self._general[ip]
                self._prune(dq_g, 60, now)
                if len(dq_g) >= s.api_rate_limit_per_minute:
                    log.warning("rate_limit_exceeded", tier="api", client_ip=ip, path=path)
                    rid = getattr(request.state, "request_id", None)
                    return JSONResponse(
                        status_code=429,
                        content={
                            "detail": "Too many requests. Try again later.",
                            "request_id": rid,
                        },
                        headers={"Retry-After": "60"},
                    )
                dq_g.append(now)

        return await call_next(request)
