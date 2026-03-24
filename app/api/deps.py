from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from redis import Redis
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_db
from app.queue.redis_score_queue import FraudScoreJobQueue
from app.redis_client import get_redis
from app.services.cache_service import CacheService
from app.services.fraud_scoring import FraudScoringService
from app.services.transaction_persistence import TransactionPersistenceService
from app.services.transaction_read_service import TransactionReadService


def get_redis_client() -> Redis:
    """FastAPI dependency: shared Redis connection for the API process."""
    return get_redis()


def get_fraud_score_queue(redis_client: Redis = Depends(get_redis_client)) -> FraudScoreJobQueue:
    return FraudScoreJobQueue(redis_client, get_settings().fraud_score_queue_key)


def get_fraud_scoring_service(
    request: Request,
    redis_client: Redis = Depends(get_redis_client),
) -> FraudScoringService:
    predictor = getattr(request.app.state, "fraud_predictor", None)
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Fraud scoring is temporarily unavailable.",
                "request_id": getattr(request.state, "request_id", None),
            },
        )
    settings = get_settings()
    return FraudScoringService(
        predictor=predictor,
        cache=CacheService(redis_client),
        persistence=TransactionPersistenceService(),
        cache_ttl_seconds=settings.fraud_cache_ttl_seconds,
    )


def get_transaction_read_service() -> TransactionReadService:
    return TransactionReadService()


__all__ = [
    "get_db",
    "get_fraud_score_queue",
    "get_fraud_scoring_service",
    "get_redis_client",
    "get_transaction_read_service",
]
