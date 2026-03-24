from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ml.dataset_spec import DatasetSpec


def load_raw_dataframe(path: Path, *, sep: str | None = None) -> pd.DataFrame:
    """Load CSV (or TSV if ``sep='\\t'``). Path must exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path, sep=sep or ",")


def load_xy(
    path: Path,
    spec: DatasetSpec,
    *,
    sep: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load CSV and split into ``X``, ``y``, and resolved numeric/categorical column names."""
    df = load_raw_dataframe(path, sep=sep)
    spec.validate_present(df)
    y = df[spec.target_column]
    numeric_cols, cat_cols = spec.resolve_feature_columns(df)
    if not numeric_cols and not cat_cols:
        raise ValueError("No feature columns resolved; check DatasetSpec and CSV headers")
    X = df[numeric_cols + cat_cols].copy()
    return X, y, numeric_cols, cat_cols


def assert_binary_target(y: pd.Series, positive_label: int | str) -> None:
    unique = set(pd.unique(y))
    if len(unique) != 2:
        raise ValueError(f"Expected binary target; got {len(unique)} unique labels: {sorted(unique)}")
    if positive_label not in unique:
        raise ValueError(f"positive_label={positive_label!r} not found in target values {unique}")
