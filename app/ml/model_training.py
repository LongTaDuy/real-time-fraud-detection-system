from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.ml.artifacts import save_training_bundle
from app.ml.data_loading import assert_binary_target, load_xy
from app.ml.dataset_spec import DatasetSpec
from app.ml.eval_metrics import evaluate_pipeline
from app.ml.preprocessing import build_column_preprocessor, build_full_pipeline


def _git_short_sha() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def compute_scale_pos_weight(y: pd.Series, positive_label: int | str) -> float:
    pos = int((y == positive_label).sum())
    neg = int((y != positive_label).sum())
    if pos == 0:
        raise ValueError("No positive (fraud) samples in training target")
    return float(neg) / float(pos)


def make_classifier(kind: str, *, random_state: int, scale_pos_weight: float) -> Any:
    kind = kind.lower().strip()
    if kind == "logistic_regression":
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            random_state=random_state,
            n_jobs=-1,
        )
    if kind == "xgboost":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
        )
    raise ValueError(f"Unknown classifier kind: {kind!r} (use logistic_regression or xgboost)")


def train_and_save(
    *,
    data_path: Path,
    spec: DatasetSpec,
    output_dir: Path,
    model_kind: str,
    model_version: str,
    random_state: int,
    test_size: float,
) -> tuple[Path, Path, dict[str, Any]]:
    X, y, numeric_cols, cat_cols = load_xy(data_path, spec)
    assert_binary_target(y, spec.positive_label)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    spw = compute_scale_pos_weight(y_train, spec.positive_label)
    preprocessor = build_column_preprocessor(numeric_cols, cat_cols)
    clf = make_classifier(model_kind, random_state=random_state, scale_pos_weight=spw)
    pipeline = build_full_pipeline(preprocessor, clf)
    pipeline.fit(X_train, y_train)

    train_report = evaluate_pipeline(
        pipeline,
        X_train,
        y_train,
        positive_label=spec.positive_label,
    )
    test_report = evaluate_pipeline(
        pipeline,
        X_test,
        y_test,
        positive_label=spec.positive_label,
    )

    feature_names_out: list[str] = list(pipeline.named_steps["preprocess"].get_feature_names_out())

    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata: dict[str, Any] = {
        "model_version": model_version,
        "model_kind": model_kind,
        "created_at_utc": created_at,
        "git_sha": _git_short_sha(),
        "data_path": str(data_path.resolve()),
        "data_sha256_prefix": _file_fingerprint(data_path),
        "dataset_spec": spec.to_dict(),
        "random_state": random_state,
        "test_size": test_size,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_counts_train": {
            "positive": int((y_train == spec.positive_label).sum()),
            "negative": int((y_train != spec.positive_label).sum()),
        },
        "imbalance_handling": (
            {"method": "class_weight=balanced"}
            if model_kind == "logistic_regression"
            else {
                "method": "scale_pos_weight",
                "scale_pos_weight": spw,
            }
        ),
        "feature_columns_numeric": numeric_cols,
        "feature_columns_categorical": cat_cols,
        "feature_names_after_preprocess": feature_names_out,
        "metrics_train": train_report["metrics"],
        "metrics_test": test_report["metrics"],
    }

    slug = model_kind.replace("_", "-")
    bundle_path, meta_path = save_training_bundle(
        output_dir,
        pipeline,
        metadata,
        model_slug=slug,
        model_version=model_version,
    )
    return bundle_path, meta_path, metadata
