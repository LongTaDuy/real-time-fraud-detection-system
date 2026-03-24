from __future__ import annotations

import time
from datetime import timezone

from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml.inference import FraudPredictor
from app.models.model_prediction import ModelPrediction
from app.models.transaction import Transaction
from app.observability import get_logger
from app.schemas.fraud import FraudScoreRequest, FraudScoreResponse
from app.services.cache_service import CacheService
from app.services.errors import (
    DuplicateTransactionError,
    FraudModelInputError,
    TransactionPayloadMismatchError,
)
from app.services.transaction_persistence import TransactionPersistenceService

log = get_logger(__name__)


def _transacted_at_equal(a, b) -> bool:
    return a.astimezone(timezone.utc) == b.astimezone(timezone.utc)


class FraudScoringService:
    """Orchestrates Redis cache, ML inference, Postgres persistence, and cache write.

    **Session contract:** the caller supplies the SQLAlchemy ``Session``; this service
    performs a single ``commit()`` on success. On failure after any ORM flush, the
    session is rolled back (except for :class:`DuplicateTransactionError`, where the
    persistence layer has already rolled back the failed flush).

    **Cache vs ``transaction_id``:** Redis is only used after the database confirms there
    is no conflicting row for the same ``transaction_id`` (see
    :class:`TransactionPayloadMismatchError`), or after confirming the persisted row
    matches the request payload for replay paths.
    """

    def __init__(
        self,
        *,
        predictor: FraudPredictor,
        persistence: TransactionPersistenceService | None = None,
        cache: CacheService | None = None,
        cache_ttl_seconds: int | None = None,
    ) -> None:
        self._predictor = predictor
        self._persistence = persistence or TransactionPersistenceService()
        self._cache = cache
        self._cache_ttl_seconds = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else get_settings().fraud_cache_ttl_seconds
        )

    @staticmethod
    def _stored_row_matches_payload(tx: Transaction, payload: FraudScoreRequest) -> bool:
        if tx.customer_id != payload.customer_id:
            return False
        if tx.amount != payload.amount:
            return False
        if tx.merchant != payload.merchant:
            return False
        if tx.category != payload.category:
            return False
        if tx.country != payload.country:
            return False
        if tx.city != payload.city:
            return False
        if tx.device_type != payload.device_type:
            return False
        if tx.ip_address != payload.ip_address:
            return False
        return _transacted_at_equal(tx.transacted_at, payload.transacted_at)

    def _response_from_prediction(self, tx: Transaction, pred: ModelPrediction) -> FraudScoreResponse:
        is_fraud = (
            tx.is_fraud_predicted
            if tx.is_fraud_predicted is not None
            else (pred.prediction_label.lower() == "fraud")
        )
        return FraudScoreResponse(
            transaction_internal_id=tx.id,
            transaction_id=tx.transaction_id,
            prediction_id=pred.id,
            fraud_score=pred.prediction_score,
            prediction_label=pred.prediction_label,
            model_version=tx.model_version or "unknown",
            is_fraud_predicted=is_fraud,
            cached=True,
            latency_ms=0,
            top_risk_factors=pred.top_risk_factors,
            scored_at=pred.created_at,
        )

    def _try_cache_then_db_replay(
        self,
        db: Session,
        existing: Transaction,
        *,
        business_transaction_id: str,
    ) -> FraudScoreResponse | None:
        if self._cache is not None:
            t0 = time.perf_counter()
            cached = self._cache.get_score(business_transaction_id)
            if cached is not None:
                latency_ms = max(0, int((time.perf_counter() - t0) * 1000))
                return cached.model_copy(update={"latency_ms": latency_ms, "cached": True})

        preds = self._persistence.list_predictions_newest_first(db, existing.id)
        if not preds:
            return None

        response = self._response_from_prediction(existing, preds[0])
        if self._cache is not None:
            self._cache.set_score(response, ttl_seconds=self._cache_ttl_seconds)
        return response

    def score_transaction(self, db: Session, payload: FraudScoreRequest) -> FraudScoreResponse:
        existing = self._persistence.get_by_business_id(db, payload.transaction_id)
        if existing is not None:
            if not self._stored_row_matches_payload(existing, payload):
                raise TransactionPayloadMismatchError(payload.transaction_id)
            replayed = self._try_cache_then_db_replay(
                db,
                existing,
                business_transaction_id=payload.transaction_id,
            )
            if replayed is not None:
                return replayed
        elif self._cache is not None:
            t0 = time.perf_counter()
            cached = self._cache.get_score(payload.transaction_id)
            if cached is not None:
                latency_ms = max(0, int((time.perf_counter() - t0) * 1000))
                return cached.model_copy(update={"latency_ms": latency_ms, "cached": True})

        started = time.perf_counter()
        try:
            inf = self._predictor.predict(
                payload,
                model_features=payload.model_features,
            )
        except ValueError as exc:
            raise FraudModelInputError(str(exc)) from exc

        try:
            if existing is not None:
                tx = existing
            else:
                tx = self._persistence.create_transaction(db, payload)

            pred = self._persistence.add_prediction(
                db,
                transaction_db_id=tx.id,
                prediction_label=inf.prediction_label,
                prediction_score=inf.fraud_score,
                top_risk_factors=inf.risk_explanations,
                latency_ms=None,
            )

            latency_ms = max(0, int((time.perf_counter() - started) * 1000))
            pred.latency_ms = latency_ms

            self._persistence.update_transaction_score_summary(
                db,
                tx,
                is_fraud_predicted=inf.is_fraud_predicted,
                fraud_score=inf.fraud_score,
                model_version=inf.model_version,
            )

            db.commit()
            db.refresh(tx)
            db.refresh(pred)
        except DuplicateTransactionError:
            raise
        except Exception:
            db.rollback()
            raise

        response = FraudScoreResponse(
            transaction_internal_id=tx.id,
            transaction_id=tx.transaction_id,
            prediction_id=pred.id,
            fraud_score=inf.fraud_score,
            prediction_label=inf.prediction_label,
            model_version=inf.model_version,
            is_fraud_predicted=inf.is_fraud_predicted,
            cached=False,
            latency_ms=latency_ms,
            top_risk_factors=inf.risk_explanations,
            scored_at=pred.created_at,
        )

        if self._cache is not None:
            self._cache.set_score(
                response,
                ttl_seconds=self._cache_ttl_seconds,
            )

        log.info(
            "fraud_score_completed",
            transaction_id=response.transaction_id,
            latency_ms=response.latency_ms,
            cached=False,
            fraud_score=response.fraud_score,
            prediction_label=response.prediction_label,
            model_version=response.model_version,
        )

        return response
