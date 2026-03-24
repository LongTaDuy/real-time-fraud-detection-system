from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, field_serializer

from app.schemas.pagination import PaginationMeta
from app.schemas.prediction import PredictionRecord
from app.schemas.transaction import TransactionResponse

if TYPE_CHECKING:
    from app.models.transaction import Transaction


class TransactionSummary(BaseModel):
    """Transaction row for list views (includes latest prediction label when present)."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    transaction_id: str
    customer_id: str
    amount: Decimal
    merchant: str | None
    country: str | None
    fraud_score: float | None
    is_fraud_predicted: bool | None
    model_version: str | None
    latest_prediction_label: str | None = None
    created_at: datetime

    @field_serializer("amount")
    def serialize_amount(self, value: Decimal) -> str:
        return format(value, "f")

    @classmethod
    def from_list_row(cls, tx: Transaction, latest_prediction_label: str | None) -> TransactionSummary:
        """Build a list-row DTO from an ORM transaction plus optional latest label subquery."""
        return cls(
            id=tx.id,
            transaction_id=tx.transaction_id,
            customer_id=tx.customer_id,
            amount=tx.amount,
            merchant=tx.merchant,
            country=tx.country,
            fraud_score=tx.fraud_score,
            is_fraud_predicted=tx.is_fraud_predicted,
            model_version=tx.model_version,
            latest_prediction_label=latest_prediction_label,
            created_at=tx.created_at,
        )


class TransactionListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[TransactionSummary]
    pagination: PaginationMeta


class TransactionDetailResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transaction: TransactionResponse
    predictions: list[PredictionRecord]
