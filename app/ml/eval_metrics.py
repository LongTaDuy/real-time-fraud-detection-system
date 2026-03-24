from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline


def y_to_binary(y: pd.Series, positive_label: int | str) -> np.ndarray:
    return (y == positive_label).astype(np.int32).values


def positive_class_proba(pipeline: Pipeline, X: pd.DataFrame, positive_label: int | str) -> np.ndarray:
    """Probability of the fraud (positive) class, aligned with ``positive_label``."""
    clf = pipeline.named_steps["classifier"]
    proba = pipeline.predict_proba(X)
    classes = list(getattr(clf, "classes_", []))
    if len(classes) < 2:
        raise ValueError("Classifier must expose binary classes_")

    idx: int | None = None
    for i, c in enumerate(classes):
        if c == positive_label or str(c) == str(positive_label):
            idx = i
            break
    if idx is None:
        idx = int(np.argmax(classes))
    return proba[:, idx]


def compute_binary_metrics(
    y_true_bin: np.ndarray,
    y_pred_bin: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average="binary",
        zero_division=0,
    )
    try:
        roc = roc_auc_score(y_true_bin, y_score)
    except ValueError:
        roc = float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc),
    }


def evaluate_pipeline(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    positive_label: int | str,
) -> dict[str, Any]:
    y_bin = y_to_binary(y, positive_label)
    y_score = positive_class_proba(pipeline, X, positive_label)
    y_pred_bin = (y_score >= 0.5).astype(np.int32)
    metrics = compute_binary_metrics(y_bin, y_pred_bin, y_score)
    return {
        "metrics": metrics,
        "n_samples": int(len(y)),
        "positive_rate": float(y_bin.mean()),
    }
