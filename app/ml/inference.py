from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.config import Settings
from app.ml.artifacts import load_training_bundle
from app.ml.explainability import build_risk_explanation_payload, build_stub_explanation_payload
from app.observability import get_logger
from app.schemas.common import MAX_AMOUNT
from app.schemas.fraud import FraudScoreRequest

log = get_logger(__name__)


@dataclass
class FraudInferenceResult:
    fraud_score: float
    prediction_label: str
    is_fraud_predicted: bool
    model_version: str
    risk_explanations: dict[str, Any]


class FraudPredictor:
    """Loads a training bundle for inference or falls back to a deterministic stub."""

    def __init__(
        self,
        *,
        pipeline: Pipeline | None,
        metadata: dict[str, Any] | None,
        stub: bool,
        fallback_model_version: str = "stub-v1",
    ) -> None:
        self._pipeline = pipeline
        self._metadata = metadata or {}
        self._stub = stub
        self._fallback_model_version = fallback_model_version
        self._positive_label: int | str = 1
        if not self._stub and self._metadata.get("dataset_spec"):
            self._positive_label = self._metadata["dataset_spec"].get("positive_label", 1)

    @property
    def is_stub(self) -> bool:
        return self._stub

    @property
    def reportable_model_version(self) -> str:
        """Version string for logs (stub fallback label or bundle metadata)."""
        if self._stub:
            return str(self._fallback_model_version)
        return str(self._metadata.get("model_version") or "unknown")

    @classmethod
    def load(cls, settings: Settings) -> FraudPredictor:
        path = settings.model_bundle_path
        if not path:
            log.warning("model_bundle_missing", detail="MODEL_BUNDLE_PATH not set; using stub")
            return cls(
                pipeline=None,
                metadata=None,
                stub=True,
                fallback_model_version=settings.default_model_version,
            )
        p = Path(path)
        if not p.is_file():
            log.warning("model_bundle_missing", path=str(p), detail="file not found; using stub")
            return cls(
                pipeline=None,
                metadata=None,
                stub=True,
                fallback_model_version=settings.default_model_version,
            )
        try:
            pipeline, metadata = load_training_bundle(p)
            return cls(
                pipeline=pipeline,
                metadata=metadata,
                stub=False,
                fallback_model_version=settings.default_model_version,
            )
        except Exception:
            log.exception("model_bundle_load_failed", path=str(p), detail="using stub predictor")
            return cls(
                pipeline=None,
                metadata=None,
                stub=True,
                fallback_model_version=settings.default_model_version,
            )

    def _stub_infer(self, payload: FraudScoreRequest) -> FraudInferenceResult:
        raw = f"{payload.transaction_id}|{payload.amount}|{payload.customer_id}"
        digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
        h = int(digest, 16)
        score = h / 0xFFFFFFFF
        is_fraud = score >= 0.85
        label = "fraud" if is_fraud else "legit"
        amount_norm = float(payload.amount) / float(MAX_AMOUNT)
        explanations = build_stub_explanation_payload(
            fraud_score=float(score),
            is_fraud=is_fraud,
            hash_component=float(score),
            amount_normalized=amount_norm,
        )
        return FraudInferenceResult(
            fraud_score=float(score),
            prediction_label=label,
            is_fraud_predicted=is_fraud,
            model_version=self._fallback_model_version,
            risk_explanations=explanations,
        )

    def _feature_columns(self) -> tuple[list[str], list[str]]:
        meta = self._metadata
        num = list(meta.get("feature_columns_numeric") or [])
        cat = list(meta.get("feature_columns_categorical") or [])
        return num, cat

    def _build_feature_row(
        self,
        payload: FraudScoreRequest,
        model_features: dict[str, Any] | None,
    ) -> pd.DataFrame:
        mf = dict(model_features or {})
        numeric_cols, cat_cols = self._feature_columns()
        row: dict[str, Any] = {}
        missing_numeric: list[str] = []

        for col in numeric_cols:
            if col in mf:
                row[col] = float(mf[col])
            elif col == "Amount":
                row[col] = float(payload.amount)
            elif col.lower() == "amount":
                row[col] = float(payload.amount)
            else:
                missing_numeric.append(col)

        if missing_numeric:
            raise ValueError(
                "Missing numeric model features (provide via `model_features`): "
                + ", ".join(missing_numeric)
            )

        attr_by_name = {
            "merchant": payload.merchant,
            "category": payload.category,
            "country": payload.country,
            "city": payload.city,
            "device_type": payload.device_type,
        }
        for col in cat_cols:
            if col in mf:
                row[col] = str(mf[col])
                continue
            val = attr_by_name.get(col)
            if val is None:
                for k, v in attr_by_name.items():
                    if k.lower() == col.lower():
                        val = v
                        break
            row[col] = "" if val is None else str(val)

        ordered = numeric_cols + cat_cols
        frame = pd.DataFrame([{c: row[c] for c in ordered}])
        return frame

    def _positive_proba(self, pipeline: Pipeline, X: pd.DataFrame) -> float:
        clf = pipeline.named_steps["classifier"]
        proba = pipeline.predict_proba(X)
        classes = list(getattr(clf, "classes_", []))
        if len(classes) < 2:
            raise ValueError("Classifier is not binary")
        idx: int | None = None
        for i, c in enumerate(classes):
            if c == self._positive_label or str(c) == str(self._positive_label):
                idx = i
                break
        if idx is None:
            idx = int(np.argmax(classes))
        return float(proba[0, idx])

    def predict(
        self,
        payload: FraudScoreRequest,
        *,
        model_features: dict[str, Any] | None = None,
    ) -> FraudInferenceResult:
        if self._stub or self._pipeline is None:
            return self._stub_infer(payload)

        pipeline = self._pipeline
        X = self._build_feature_row(payload, model_features)
        score = self._positive_proba(pipeline, X)
        is_fraud = score >= 0.5
        label = "fraud" if is_fraud else "legit"
        explanations = build_risk_explanation_payload(
            pipeline,
            X,
            fraud_score=score,
            is_fraud=is_fraud,
            top_k=5,
        )
        version = str(self._metadata.get("model_version") or "unknown")
        return FraudInferenceResult(
            fraud_score=score,
            prediction_label=label,
            is_fraud_predicted=is_fraud,
            model_version=version,
            risk_explanations=explanations,
        )
