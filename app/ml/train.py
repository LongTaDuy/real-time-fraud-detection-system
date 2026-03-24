"""Train fraud models (Logistic Regression + XGBoost) from a CSV + DatasetSpec.

Run from repository root::

    python -m app.ml.train --data-path ./data/creditcard.csv --model both
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from app.ml.dataset_spec import DatasetSpec
from app.ml.model_training import _git_short_sha, train_and_save


def _default_model_version() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = _git_short_sha() or "nogit"
    return f"{ts}_{sha}"


def _resolve_spec(spec_path: Path | None) -> DatasetSpec:
    if spec_path is not None:
        return DatasetSpec.from_json_file(Path(spec_path))
    packaged = Path(__file__).resolve().parent / "specs" / "creditcard.json"
    if packaged.is_file():
        return DatasetSpec.from_json_file(packaged)
    return DatasetSpec.creditcard_default()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to CSV (e.g. Kaggle creditcard.csv).",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=None,
        help="JSON DatasetSpec; default packaged creditcard.json or built-in default.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("app/ml/artifacts"),
        help="Directory for bundle.joblib + metadata.json",
    )
    parser.add_argument(
        "--model",
        choices=["logistic_regression", "xgboost", "both"],
        default="both",
        help="Which classifier to train.",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Semantic or snapshot version stored in metadata; default UTC+git.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    spec = _resolve_spec(args.spec_path)
    version = args.model_version or _default_model_version()
    output_dir = Path(args.output_dir)

    models: list[str]
    if args.model == "both":
        models = ["logistic_regression", "xgboost"]
    else:
        models = [args.model]

    summaries: list[dict] = []
    for kind in models:
        bundle_path, meta_path, metadata = train_and_save(
            data_path=args.data_path,
            spec=spec,
            output_dir=output_dir,
            model_kind=kind,
            model_version=version,
            random_state=args.random_state,
            test_size=args.test_size,
        )
        summaries.append(
            {
                "model_kind": kind,
                "bundle_path": str(bundle_path),
                "metadata_path": str(meta_path),
                "metrics_test": metadata["metrics_test"],
            }
        )

    print(json.dumps({"model_version": version, "runs": summaries}, indent=2))


if __name__ == "__main__":
    main()
