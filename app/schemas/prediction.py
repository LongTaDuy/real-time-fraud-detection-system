from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.pagination import PaginationMeta

if TYPE_CHECKING:
    from app.models.model_prediction import ModelPrediction
    from app.models.transaction import Transaction


class PredictionRecord(BaseModel):
    """One stored model prediction row."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    prediction_label: str
    prediction_score: float = Field(ge=0.0, le=1.0)
    top_risk_factors: dict[str, Any] | list[Any] | None
    latency_ms: int | None = Field(default=None, ge=0)
    created_at: datetime

    @classmethod
    def from_prediction(cls, row: object) -> "PredictionRecord":
        return cls.model_validate(row)


class PredictionHistoryResponse(BaseModel):
    """All predictions recorded for a transaction (newest first)."""

    model_config = ConfigDict(from_attributes=True)

    transaction_internal_id: uuid.UUID
    transaction_id: str
    predictions: list[PredictionRecord]


class RecentPredictionItem(BaseModel):
    """One prediction in the global recent feed."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    transaction_internal_id: uuid.UUID
    transaction_id: str
    prediction_label: str
    prediction_score: float = Field(ge=0.0, le=1.0)
    latency_ms: int | None = Field(default=None, ge=0)
    created_at: datetime
    top_risk_factors: dict[str, Any] | list[Any] | None = None

    @classmethod
    def from_prediction_and_transaction(
        cls,
        pred: ModelPrediction,
        tx: Transaction,
    ) -> RecentPredictionItem:
        return cls(
            id=pred.id,
            transaction_internal_id=tx.id,
            transaction_id=tx.transaction_id,
            prediction_label=pred.prediction_label,
            prediction_score=pred.prediction_score,
            latency_ms=pred.latency_ms,
            created_at=pred.created_at,
            top_risk_factors=pred.top_risk_factors,
        )


class RecentPredictionsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RecentPredictionItem]
    pagination: PaginationMeta
