"""Evaluate a saved training bundle on a holdout CSV (same schema as training).

Run from repository root::

    python -m app.ml.evaluate --bundle app/ml/artifacts/xgboost_*_bundle.joblib --data-path ./data/holdout.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.ml.artifacts import load_training_bundle
from app.ml.data_loading import load_xy
from app.ml.dataset_spec import DatasetSpec
from app.ml.eval_metrics import evaluate_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved fraud model bundle.")
    parser.add_argument("--bundle", type=Path, required=True, help="Path to *_bundle.joblib")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="CSV with the same columns as training (including target).",
    )
    parser.add_argument(
        "--spec-path",
        type=Path,
        default=None,
        help="Optional DatasetSpec JSON; default is metadata.dataset_spec from the bundle.",
    )
    args = parser.parse_args()

    pipeline, metadata = load_training_bundle(args.bundle)
    if args.spec_path is not None:
        spec = DatasetSpec.from_json_file(args.spec_path)
    else:
        spec = DatasetSpec.from_dict(metadata["dataset_spec"])

    X, y, _, _ = load_xy(args.data_path, spec)
    report = evaluate_pipeline(
        pipeline,
        X,
        y,
        positive_label=spec.positive_label,
    )
    out = {
        "bundle": str(args.bundle.resolve()),
        "data_path": str(args.data_path.resolve()),
        "model_version": metadata.get("model_version"),
        "model_kind": metadata.get("model_kind"),
        "evaluation": report,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
