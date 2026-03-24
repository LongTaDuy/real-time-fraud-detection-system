"""Human-readable fraud explanations without per-request SHAP (MVP).

We intentionally skip SHAP as the default path: it adds a heavy dependency, often
needs a background dataset for linear models, and TreeExplainer latency can spike
on wide one-hot spaces. Instead:

- **LogisticRegression:** term ``coef_j * x_j`` is a direct additive contribution
  to the linear score (same sign as effect on log-odds for the positive class).
- **Tree ensembles (e.g. XGBoost):** use ``importance_j * |x_j|`` as a cheap,
  direction-agnostic influence signal—good enough for short “what stood out” copy.

For production SHAP, add a background sample and optional ``shap`` in a later
iteration behind a feature flag.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

Direction = Literal["increases_fraud_risk", "decreases_fraud_risk", "notable"]


def humanize_feature_label(raw_name: str) -> str:
    name = str(raw_name)
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
            break
    return name.replace("_", " ").strip().title() or str(raw_name)


def _contributions(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, str]:
    pre = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["classifier"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    values = np.asarray(Xt, dtype=float).ravel()
    names = np.asarray(pre.get_feature_names_out(), dtype=object)

    if hasattr(clf, "coef_"):
        weights = np.asarray(clf.coef_, dtype=float).reshape(-1)
        n = int(min(names.size, weights.size, values.size))
        contrib = weights[:n] * values[:n]
        return np.asarray(names[:n], dtype=object), contrib, "linear_contribution"

    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float).reshape(-1)
        n = int(min(names.size, imp.size, values.size))
        contrib = imp[:n] * np.abs(values[:n])
        return np.asarray(names[:n], dtype=object), contrib, "tree_importance_heuristic"

    return names, np.zeros_like(values), "none"


def _direction_for_contribution(method: str, contribution: float) -> Direction:
    if method != "linear_contribution":
        return "notable"
    if contribution > 1e-6:
        return "increases_fraud_risk"
    if contribution < -1e-6:
        return "decreases_fraud_risk"
    return "notable"


def _message_for_factor(
    *,
    method: str,
    label: str,
    direction: Direction,
    contribution: float,
) -> str:
    if method == "linear_contribution":
        if direction == "increases_fraud_risk":
            return f"{label}: higher value pushes this transaction toward fraud in the linear model."
        if direction == "decreases_fraud_risk":
            return f"{label}: higher value pulls this transaction toward legitimate in the linear model."
        return f"{label}: small linear effect on the fraud score."

    if method == "tree_importance_heuristic":
        return (
            f"{label}: stood out after combining global feature importance with this input "
            f"(heuristic; not a causal statement)."
        )

    return f"{label}: included in the model input."


def _headline(
    *,
    method: str,
    fraud_score: float,
    is_fraud: bool,
    top_label: str | None,
) -> str:
    verdict = "flagged as higher fraud risk" if is_fraud else "scored as lower fraud risk"
    if top_label:
        return (
            f"Transaction {verdict} (score {fraud_score:.2f}). "
            f"Largest signal: {top_label}."
        )
    return f"Transaction {verdict} (score {fraud_score:.2f})."


def build_risk_explanation_payload(
    pipeline: Pipeline,
    X: pd.DataFrame,
    *,
    fraud_score: float,
    is_fraud: bool,
    top_k: int = 5,
) -> dict[str, Any]:
    names, contrib, method = _contributions(pipeline, X)
    if method == "none" or contrib.size == 0:
        return {
            "method": "unavailable",
            "headline": _headline(
                method="unavailable",
                fraud_score=fraud_score,
                is_fraud=is_fraud,
                top_label=None,
            ),
            "factors": [],
        }

    order = np.argsort(-np.abs(contrib))[:top_k]
    max_abs = float(np.max(np.abs(contrib[order]))) or 1.0

    factors: list[dict[str, Any]] = []
    top_label: str | None = None
    for idx in order:
        c = float(contrib[idx])
        raw = str(names[idx])
        label = humanize_feature_label(raw)
        if top_label is None:
            top_label = label
        direction = _direction_for_contribution(method, c)
        strength = min(1.0, max(0.0, abs(c) / max_abs))
        factors.append(
            {
                "feature": raw,
                "label": label,
                "direction": direction,
                "strength": round(strength, 4),
                "contribution": round(c, 6),
                "message": _message_for_factor(
                    method=method,
                    label=label,
                    direction=direction,
                    contribution=c,
                ),
            }
        )

    return {
        "method": method,
        "headline": _headline(
            method=method,
            fraud_score=fraud_score,
            is_fraud=is_fraud,
            top_label=top_label,
        ),
        "factors": factors,
    }


def build_stub_explanation_payload(
    *,
    fraud_score: float,
    is_fraud: bool,
    hash_component: float,
    amount_normalized: float,
) -> dict[str, Any]:
    verdict = "flagged as higher fraud risk" if is_fraud else "scored as lower fraud risk"
    return {
        "method": "stub",
        "headline": (
            f"Demo mode (no ML bundle): transaction {verdict} (score {fraud_score:.2f}). "
            "Load MODEL_BUNDLE_PATH for real model explanations."
        ),
        "factors": [
            {
                "feature": "demo_hash",
                "label": "Demo hash signal",
                "direction": "notable",
                "strength": 1.0,
                "contribution": round(hash_component, 6),
                "message": "Deterministic hash of ids/amount for portfolio demos—not a learned signal.",
            },
            {
                "feature": "amount_normalized",
                "label": "Amount (normalized)",
                "direction": "notable",
                "strength": min(1.0, float(amount_normalized) * 50 + 0.1),
                "contribution": round(amount_normalized, 6),
                "message": "Relative transaction size vs. a reference maximum, for illustration only.",
            },
        ],
    }
