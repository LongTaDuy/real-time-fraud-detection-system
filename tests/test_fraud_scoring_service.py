from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from app.ml.inference import FraudInferenceResult
from app.schemas.fraud import FraudScoreRequest, FraudScoreResponse
from app.services.cache_service import CacheService
from app.services.errors import FraudModelInputError, TransactionPayloadMismatchError
from app.services.fraud_scoring import FraudScoringService


@pytest.fixture
def sample_request() -> FraudScoreRequest:
    return FraudScoreRequest(
        transaction_id="txn_unit_svc",
        customer_id="cus_unit",
        amount=Decimal("99.00"),
        merchant="UnitMart",
        category="retail",
        country="US",
        city="Portland",
        device_type="mobile",
        ip_address="192.0.2.50",
        transacted_at=datetime(2025, 2, 1, 10, 0, 0, tzinfo=timezone.utc),
    )


def test_cache_hit_returns_without_predictor_or_persistence(
    sample_request: FraudScoreRequest,
) -> None:
    cached = FraudScoreResponse(
        transaction_internal_id=uuid.uuid4(),
        transaction_id=sample_request.transaction_id,
        prediction_id=uuid.uuid4(),
        fraud_score=0.11,
        prediction_label="legit",
        model_version="cached-v",
        is_fraud_predicted=False,
        cached=True,
        latency_ms=1,
        top_risk_factors={"method": "test"},
        scored_at=datetime.now(timezone.utc),
    )
    cache = MagicMock(spec=CacheService)
    cache.get_score.return_value = cached
    predictor = MagicMock()
    persistence = MagicMock()
    persistence.get_by_business_id.return_value = None

    svc = FraudScoringService(predictor=predictor, cache=cache, persistence=persistence)
    db = MagicMock()
    out = svc.score_transaction(db, sample_request)

    assert out.cached is True
    assert out.fraud_score == cached.fraud_score
    predictor.predict.assert_not_called()
    persistence.create_transaction.assert_not_called()
    db.commit.assert_not_called()


def test_existing_transaction_payload_mismatch_raises(
    sample_request: FraudScoreRequest,
) -> None:
    existing = MagicMock()
    existing.customer_id = "someone_else"
    persistence = MagicMock()
    persistence.get_by_business_id.return_value = existing
    svc = FraudScoringService(
        predictor=MagicMock(),
        cache=MagicMock(spec=CacheService),
        persistence=persistence,
    )
    with pytest.raises(TransactionPayloadMismatchError) as exc_info:
        svc.score_transaction(MagicMock(), sample_request)
    assert exc_info.value.transaction_id == sample_request.transaction_id


def test_predictor_valueerror_maps_to_fraud_model_input_error(
    sample_request: FraudScoreRequest,
) -> None:
    cache = MagicMock(spec=CacheService)
    cache.get_score.return_value = None
    predictor = MagicMock()
    predictor.predict.side_effect = ValueError("Missing numeric model features: V9")
    persistence = MagicMock()
    persistence.get_by_business_id.return_value = None

    svc = FraudScoringService(predictor=predictor, cache=cache, persistence=persistence)
    with pytest.raises(FraudModelInputError, match="V9"):
        svc.score_transaction(MagicMock(), sample_request)


def test_successful_score_commits_and_sets_cache(
    sqlite_engine,
    sample_request: FraudScoreRequest,
    redis_client,
) -> None:
    from sqlalchemy.orm import sessionmaker

    from app.ml.inference import FraudPredictor
    from app.services.transaction_persistence import TransactionPersistenceService

    Session = sessionmaker(bind=sqlite_engine)
    db = Session()
    try:
        inf = FraudInferenceResult(
            fraud_score=0.2,
            prediction_label="legit",
            is_fraud_predicted=False,
            model_version="stub-v1",
            risk_explanations={"method": "stub", "headline": "x", "factors": []},
        )
        predictor = MagicMock(spec=FraudPredictor)
        predictor.predict.return_value = inf

        cache = CacheService(redis_client)
        svc = FraudScoringService(
            predictor=predictor,
            cache=cache,
            persistence=TransactionPersistenceService(),
        )
        out = svc.score_transaction(db, sample_request)

        assert out.cached is False
        assert out.fraud_score == 0.2
        predictor.predict.assert_called_once()

        cached_read = cache.get_score(sample_request.transaction_id)
        assert cached_read is not None
        assert cached_read.fraud_score == 0.2
    finally:
        db.close()
