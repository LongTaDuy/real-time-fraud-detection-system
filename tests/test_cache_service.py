from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.schemas.fraud import FraudScoreResponse
from app.services.cache_service import CacheService


def _sample_response(*, transaction_id: str = "txn_cache_1") -> FraudScoreResponse:
    now = datetime.now(timezone.utc)
    return FraudScoreResponse(
        transaction_internal_id=uuid.uuid4(),
        transaction_id=transaction_id,
        prediction_id=uuid.uuid4(),
        fraud_score=0.42,
        prediction_label="fraud",
        model_version="stub-v1",
        is_fraud_predicted=True,
        cached=False,
        latency_ms=33,
        top_risk_factors={"method": "stub", "headline": "test", "factors": []},
        scored_at=now,
    )


def test_cache_roundtrip(redis_client) -> None:
    svc = CacheService(redis_client)
    assert svc.get_score("missing") is None

    original = _sample_response()
    svc.set_score(original, ttl_seconds=60)
    cached = svc.get_score(original.transaction_id)
    assert cached is not None
    assert cached.transaction_id == original.transaction_id
    assert cached.fraud_score == original.fraud_score
    assert cached.prediction_label == original.prediction_label
    assert cached.cached is True
    assert cached.top_risk_factors == original.top_risk_factors


def test_cache_isolated_per_business_id(redis_client) -> None:
    svc = CacheService(redis_client)
    a = _sample_response(transaction_id="txn_a")
    b = _sample_response(transaction_id="txn_b")
    svc.set_score(a, ttl_seconds=60)
    svc.set_score(b, ttl_seconds=60)
    hit_a = svc.get_score("txn_a")
    hit_b = svc.get_score("txn_b")
    assert hit_a is not None and hit_b is not None
    assert hit_a.fraud_score == a.fraud_score
    assert hit_b.fraud_score == b.fraud_score


def test_cache_invalid_json_raises(redis_client) -> None:
    from pydantic import ValidationError

    redis_client.set("fraud:score:v1:tx:txn_bad_json", "not-valid-json{")
    svc = CacheService(redis_client)
    with pytest.raises(ValidationError):
        svc.get_score("txn_bad_json")
