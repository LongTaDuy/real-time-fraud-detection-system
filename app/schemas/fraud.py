from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from pydantic import (
    AliasChoices,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.schemas.common import (
    ExternalId,
    OptionalCountry,
    OptionalIP,
    StrictAmount,
    _normalize_optional_str,
)

_MODEL_FEATURE_KEY = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class FraudScoreRequest(BaseModel):
    """Inbound payload to score a single transaction (API → service)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    transaction_id: ExternalId
    customer_id: str = Field(min_length=1, max_length=128)
    amount: StrictAmount
    merchant: str | None = Field(default=None, max_length=255)
    category: str | None = Field(default=None, max_length=128)
    country: OptionalCountry = None
    city: str | None = Field(default=None, max_length=128)
    device_type: str | None = Field(default=None, max_length=64)
    ip_address: OptionalIP = None
    transacted_at: AwareDatetime = Field(
        validation_alias=AliasChoices("transacted_at", "timestamp"),
    )
    model_features: dict[str, float | str] | None = Field(
        default=None,
        description="Extra columns expected by the trained tabular model (e.g. V1..V28). "
        "Required numeric columns from training must appear here unless mapped (e.g. Amount).",
    )

    @field_validator("model_features")
    @classmethod
    def model_features_valid(cls, v: dict[str, float | str] | None) -> dict[str, float | str] | None:
        if v is None:
            return None
        if len(v) > 400:
            raise ValueError("model_features may contain at most 400 keys")
        for key in v:
            if not _MODEL_FEATURE_KEY.fullmatch(key):
                raise ValueError(
                    "model_features keys must be 1-64 characters matching [A-Za-z0-9_.-]",
                )
        return v

    @field_validator("customer_id", mode="before")
    @classmethod
    def customer_id_clean(cls, v: object) -> object:
        if isinstance(v, str):
            s = v.strip()
            if not s:
                raise ValueError("customer_id must not be blank")
            if any(ord(c) < 32 or ord(c) == 127 for c in s):
                raise ValueError("customer_id must not contain control characters")
            return s
        return v

    @model_validator(mode="after")
    def transacted_at_reasonable_bounds(self) -> FraudScoreRequest:
        ts = self.transacted_at
        if ts.tzinfo is None:
            raise ValueError("transacted_at must be timezone-aware")
        ts_utc = ts.astimezone(timezone.utc)
        now_utc = datetime.now(timezone.utc)
        if ts_utc > now_utc + timedelta(days=1):
            raise ValueError("transacted_at must not be more than one day in the future")
        if ts_utc < datetime(2000, 1, 1, tzinfo=timezone.utc):
            raise ValueError("transacted_at must be on or after 2000-01-01 UTC")
        return self

    @field_validator("merchant", "category", "city", "device_type", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: object) -> object:
        if isinstance(v, str):
            return _normalize_optional_str(v)
        return v


class FraudScoreAsyncAcceptedResponse(BaseModel):
    """Acknowledgement for async enqueue-only scoring."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted"] = "accepted"
    transaction_id: str
    message: str = "Transaction queued for background fraud scoring."


class FraudScoreResponse(BaseModel):
    """Result of fraud scoring including persistence identifiers."""

    model_config = ConfigDict(from_attributes=True)

    transaction_internal_id: uuid.UUID
    transaction_id: str
    prediction_id: uuid.UUID
    fraud_score: float = Field(ge=0.0, le=1.0)
    prediction_label: str = Field(min_length=1, max_length=32)
    model_version: str = Field(min_length=1, max_length=64)
    is_fraud_predicted: bool
    cached: bool
    latency_ms: int | None = Field(default=None, ge=0)
    top_risk_factors: dict[str, Any] | list[Any] | None = None
    scored_at: datetime


class RiskFactorItem(BaseModel):
    """One human-readable row in a fraud explanation."""

    model_config = ConfigDict(extra="ignore")

    label: str
    message: str
    direction: str = "notable"
    strength: float = Field(ge=0.0, le=1.0)
    feature: str | None = None
    contribution: float | None = None


class RiskExplanationPayload(BaseModel):
    """Structured explainability payload stored in ``model_predictions.top_risk_factors``."""

    model_config = ConfigDict(extra="forbid")

    method: str
    headline: str
    factors: list[RiskFactorItem]


class FraudScoreApiResponse(BaseModel):
    """HTTP-facing score payload (aliases for API clarity)."""

    fraud_score: float = Field(ge=0.0, le=1.0)
    predicted_label: str
    model_version: str
    risk_explanations: RiskExplanationPayload | None = None
    transaction_internal_id: uuid.UUID
    transaction_id: str
    prediction_id: uuid.UUID
    latency_ms: int = Field(ge=0)
    cached: bool
    scored_at: datetime

    @classmethod
    def from_domain(cls, row: FraudScoreResponse) -> "FraudScoreApiResponse":
        raw = row.top_risk_factors
        expl: RiskExplanationPayload | None = None
        if (
            isinstance(raw, dict)
            and isinstance(raw.get("factors"), list)
            and isinstance(raw.get("headline"), str)
        ):
            expl = RiskExplanationPayload.model_validate(raw)
        elif isinstance(raw, dict) and raw:
            legacy_factors: list[RiskFactorItem] = []
            for k, v in raw.items():
                if isinstance(v, (int, float)):
                    fv = float(v)
                    legacy_factors.append(
                        RiskFactorItem(
                            label=str(k),
                            message=f"Legacy cached numeric signal: {k}={fv:.4f}",
                            direction="notable",
                            strength=min(1.0, abs(fv)),
                            contribution=fv,
                        )
                    )
            if legacy_factors:
                expl = RiskExplanationPayload(
                    method="legacy_cache",
                    headline="Cached explanation stored as a flat numeric map (older format).",
                    factors=legacy_factors[:8],
                )

        return cls(
            fraud_score=row.fraud_score,
            predicted_label=row.prediction_label,
            model_version=row.model_version,
            risk_explanations=expl,
            transaction_internal_id=row.transaction_internal_id,
            transaction_id=row.transaction_id,
            prediction_id=row.prediction_id,
            latency_ms=row.latency_ms or 0,
            cached=row.cached,
            scored_at=row.scored_at,
        )
