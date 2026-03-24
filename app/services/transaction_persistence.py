from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.model_prediction import ModelPrediction
from app.models.transaction import Transaction
from app.schemas.fraud import FraudScoreRequest
from app.schemas.prediction import PredictionHistoryResponse, PredictionRecord
from app.services.errors import DuplicateTransactionError


class TransactionPersistenceService:
    """Database access for transactions and related predictions (no HTTP concerns)."""

    @staticmethod
    def _transaction_row_from_request(payload: FraudScoreRequest) -> Transaction:
        return Transaction(
            transaction_id=payload.transaction_id,
            customer_id=payload.customer_id,
            amount=payload.amount,
            merchant=payload.merchant,
            category=payload.category,
            country=payload.country,
            city=payload.city,
            device_type=payload.device_type,
            ip_address=payload.ip_address,
            transacted_at=payload.transacted_at,
        )

    def create_transaction(self, db: Session, payload: FraudScoreRequest) -> Transaction:
        row = self._transaction_row_from_request(payload)
        db.add(row)
        try:
            db.flush()
        except IntegrityError as exc:
            db.rollback()
            raise DuplicateTransactionError(payload.transaction_id) from exc
        return row

    def update_transaction_score_summary(
        self,
        db: Session,
        row: Transaction,
        *,
        is_fraud_predicted: bool,
        fraud_score: float,
        model_version: str,
    ) -> None:
        row.is_fraud_predicted = is_fraud_predicted
        row.fraud_score = fraud_score
        row.model_version = model_version

    def add_prediction(
        self,
        db: Session,
        *,
        transaction_db_id: uuid.UUID,
        prediction_label: str,
        prediction_score: float,
        top_risk_factors: dict[str, Any] | list[Any] | None,
        latency_ms: int | None,
    ) -> ModelPrediction:
        pred = ModelPrediction(
            transaction_id=transaction_db_id,
            prediction_label=prediction_label,
            prediction_score=prediction_score,
            top_risk_factors=top_risk_factors,
            latency_ms=latency_ms,
        )
        db.add(pred)
        db.flush()
        return pred

    def get_by_internal_id(self, db: Session, internal_id: uuid.UUID) -> Transaction | None:
        return db.get(Transaction, internal_id)

    def get_by_business_id(self, db: Session, business_transaction_id: str) -> Transaction | None:
        stmt = select(Transaction).where(Transaction.transaction_id == business_transaction_id)
        return db.scalars(stmt).first()

    def list_predictions_newest_first(
        self,
        db: Session,
        transaction_db_id: uuid.UUID,
    ) -> list[ModelPrediction]:
        stmt = (
            select(ModelPrediction)
            .where(ModelPrediction.transaction_id == transaction_db_id)
            .order_by(ModelPrediction.created_at.desc())
        )
        return list(db.scalars(stmt).all())

    def build_prediction_history(
        self,
        db: Session,
        *,
        transaction: Transaction,
    ) -> PredictionHistoryResponse:
        preds = self.list_predictions_newest_first(db, transaction.id)
        return PredictionHistoryResponse(
            transaction_internal_id=transaction.id,
            transaction_id=transaction.transaction_id,
            predictions=[PredictionRecord.from_prediction(p) for p in preds],
        )

    def get_prediction_history(
        self,
        db: Session,
        *,
        internal_id: uuid.UUID,
    ) -> PredictionHistoryResponse | None:
        tx = self.get_by_internal_id(db, internal_id)
        if tx is None:
            return None
        return self.build_prediction_history(db, transaction=tx)
