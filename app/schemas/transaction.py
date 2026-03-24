from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_serializer

if TYPE_CHECKING:
    from app.models.transaction import Transaction


class TransactionResponse(BaseModel):
    """Single transaction as returned by read APIs."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    transaction_id: str
    customer_id: str
    amount: Decimal
    merchant: str | None
    category: str | None
    country: str | None
    city: str | None
    device_type: str | None
    ip_address: str | None
    transacted_at: datetime = Field(
        serialization_alias="timestamp",
    )
    is_fraud_predicted: bool | None
    fraud_score: float | None
    model_version: str | None
    created_at: datetime

    @field_serializer("amount")
    def serialize_amount(self, value: Decimal) -> str:
        return format(value, "f")

    @classmethod
    def from_transaction(cls, row: Transaction) -> TransactionResponse:
        return cls.model_validate(row)
