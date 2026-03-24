import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict
from redis import Redis

from app.schemas.fraud import FraudScoreResponse


class _ScoreCachePayload(BaseModel):
    """Redis JSON shape for a cached fraud score (internal contract)."""

    model_config = ConfigDict(extra="forbid")

    transaction_internal_id: uuid.UUID
    transaction_id: str
    prediction_id: uuid.UUID
    fraud_score: float
    prediction_label: str
    model_version: str
    is_fraud_predicted: bool
    latency_ms: int | None
    top_risk_factors: dict[str, Any] | list[Any] | None
    scored_at: datetime


class CacheService:
    """Redis-backed cache for idempotent reads of recent fraud scores."""

    def __init__(self, redis_client: Redis, *, key_prefix: str = "fraud:score:v1") -> None:
        self._redis = redis_client
        self._prefix = key_prefix

    def _business_key(self, business_transaction_id: str) -> str:
        return f"{self._prefix}:tx:{business_transaction_id}"

    def get_score(self, business_transaction_id: str) -> FraudScoreResponse | None:
        raw = self._redis.get(self._business_key(business_transaction_id))
        if raw is None:
            return None
        payload = _ScoreCachePayload.model_validate_json(raw)
        return FraudScoreResponse(
            transaction_internal_id=payload.transaction_internal_id,
            transaction_id=payload.transaction_id,
            prediction_id=payload.prediction_id,
            fraud_score=payload.fraud_score,
            prediction_label=payload.prediction_label,
            model_version=payload.model_version,
            is_fraud_predicted=payload.is_fraud_predicted,
            cached=True,
            latency_ms=payload.latency_ms,
            top_risk_factors=payload.top_risk_factors,
            scored_at=payload.scored_at,
        )

    def set_score(
        self,
        response: FraudScoreResponse,
        *,
        ttl_seconds: int,
    ) -> None:
        payload = _ScoreCachePayload(
            transaction_internal_id=response.transaction_internal_id,
            transaction_id=response.transaction_id,
            prediction_id=response.prediction_id,
            fraud_score=response.fraud_score,
            prediction_label=response.prediction_label,
            model_version=response.model_version,
            is_fraud_predicted=response.is_fraud_predicted,
            latency_ms=response.latency_ms,
            top_risk_factors=response.top_risk_factors,
            scored_at=response.scored_at,
        )
        self._redis.setex(
            self._business_key(response.transaction_id),
            ttl_seconds,
            payload.model_dump_json(),
        )
