"""Redis list queue for async fraud scoring jobs (LPUSH / BRPOP FIFO)."""

from __future__ import annotations

from redis import Redis

from app.schemas.fraud import FraudScoreRequest


class FraudScoreJobQueue:
    """Producer + blocking consumer for JSON-serialized :class:`FraudScoreRequest` payloads."""

    def __init__(self, redis_client: Redis, queue_key: str) -> None:
        self._redis = redis_client
        self._key = queue_key

    @property
    def key(self) -> str:
        return self._key

    def enqueue(self, body: FraudScoreRequest) -> None:
        """Push a job onto the queue (left); worker pops from the right (FIFO)."""
        self._redis.lpush(self._key, body.model_dump_json())

    def blocking_pop_raw(self, *, timeout_seconds: int) -> str | None:
        """Block up to ``timeout_seconds`` waiting for a job; return JSON string or ``None``."""
        out = self._redis.brpop(self._key, timeout=timeout_seconds)
        if out is None:
            return None
        _key, raw = out
        return raw
