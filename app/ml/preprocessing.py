from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_column_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Fit-ready preprocessor: numeric (impute + scale), categorical (impute + OHE)."""
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    if numeric_features:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, numeric_features))

    if categorical_features:
        cat_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                        min_frequency=None,
                    ),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_features))

    if not transformers:
        raise ValueError("At least one of numeric_features or categorical_features must be non-empty")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_full_pipeline(preprocessor: ColumnTransformer, classifier) -> Pipeline:
    from sklearn.pipeline import Pipeline as SkPipeline

    return SkPipeline(
        steps=[
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ]
    )
