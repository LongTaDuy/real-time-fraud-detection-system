"""Process-wide Redis client (API) and helpers for constructing clients (e.g. workers)."""

from __future__ import annotations

from redis import Redis

from app.config import get_settings

_client: Redis | None = None


def build_redis_client(
    *,
    redis_url: str,
    decode_responses: bool = True,
    socket_connect_timeout: float | None = None,
    socket_timeout: float | None = None,
) -> Redis:
    """Create a new Redis client (caller owns lifecycle — use in workers and one-off scripts)."""
    kwargs: dict = {"decode_responses": decode_responses}
    if socket_connect_timeout is not None:
        kwargs["socket_connect_timeout"] = socket_connect_timeout
    if socket_timeout is not None:
        kwargs["socket_timeout"] = socket_timeout
    return Redis.from_url(redis_url, **kwargs)


def get_redis() -> Redis:
    """Singleton Redis client for the API process (lazy, tied to :func:`close_redis`)."""
    global _client
    if _client is None:
        s = get_settings()
        _client = build_redis_client(
            redis_url=s.redis_url,
            socket_connect_timeout=s.redis_socket_connect_timeout_seconds,
            socket_timeout=s.redis_socket_timeout_seconds,
        )
    return _client


def close_redis() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
