"""Client IP for rate limiting (optional ``X-Forwarded-For`` when explicitly trusted)."""

from __future__ import annotations

from starlette.requests import Request


def get_client_ip(request: Request, *, trust_proxy_headers: bool) -> str:
    if trust_proxy_headers:
        xff = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
        if xff:
            return xff.split(",")[0].strip() or "unknown"
    if request.client is not None:
        return request.client.host
    return "unknown"
