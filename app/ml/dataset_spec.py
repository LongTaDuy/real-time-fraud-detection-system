from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetSpec:
    """Declarative description of a tabular fraud CSV (easy to swap datasets).

    - ``numeric_columns`` / ``categorical_columns``: explicit lists, or leave numeric None to infer.
    - ``drop_columns``: removed from features (e.g. ``Time`` on creditcard).
    - ``positive_label``: fraud class for metrics and for deriving imbalance ratios.
    """

    target_column: str
    id_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] | None = None
    drop_columns: list[str] = field(default_factory=list)
    positive_label: int | str = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSpec:
        return cls(
            target_column=str(data["target_column"]),
            id_columns=list(data.get("id_columns", [])),
            categorical_columns=list(data.get("categorical_columns", [])),
            numeric_columns=data.get("numeric_columns"),
            drop_columns=list(data.get("drop_columns", [])),
            positive_label=data.get("positive_label", 1),
        )

    @classmethod
    def from_json_file(cls, path: Path) -> DatasetSpec:
        with path.open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def creditcard_default(cls) -> DatasetSpec:
        """ULB/Kaggle Credit Card style: ``Class`` target, numeric V* + Amount."""
        return cls(
            target_column="Class",
            id_columns=[],
            categorical_columns=[],
            numeric_columns=None,
            drop_columns=["Time"],
            positive_label=1,
        )

    def resolve_feature_columns(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        forbidden = set(self.id_columns) | {self.target_column} | set(self.drop_columns)
        remaining = [c for c in df.columns if c not in forbidden]

        if self.numeric_columns is not None:
            numeric = [c for c in self.numeric_columns if c in df.columns]
            categorical = [c for c in self.categorical_columns if c in df.columns]
            missing = set(self.numeric_columns) | set(self.categorical_columns)
            missing -= set(numeric) | set(categorical)
            if missing:
                raise ValueError(f"Spec references unknown columns: {sorted(missing)}")
            return numeric, categorical

        categorical = [c for c in self.categorical_columns if c in remaining]
        cat_set = set(categorical)
        numeric: list[str] = []
        other: list[str] = []
        for c in remaining:
            if c in cat_set:
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric.append(c)
            else:
                other.append(c)
        categorical.extend(other)
        numeric.sort()
        categorical.sort()
        return numeric, categorical

    def validate_present(self, df: pd.DataFrame) -> None:
        if self.target_column not in df.columns:
            raise ValueError(f"Missing target column {self.target_column!r} in CSV")
        for col in self.id_columns:
            if col not in df.columns:
                raise ValueError(f"Missing id column {col!r} in CSV")
        for col in self.drop_columns:
            if col not in df.columns:
                raise ValueError(f"Missing drop column {col!r} in CSV")
