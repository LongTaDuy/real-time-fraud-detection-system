from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline


def save_training_bundle(
    output_dir: Path,
    pipeline: Pipeline,
    metadata: dict[str, Any],
    *,
    model_slug: str,
    model_version: str,
) -> tuple[Path, Path]:
    """Persist fitted ``Pipeline`` (preprocess + classifier) plus sidecar JSON metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_version = model_version.replace("/", "-").replace(" ", "_")
    base = f"{model_slug}_{safe_version}"
    bundle_path = output_dir / f"{base}_bundle.joblib"
    meta_path = output_dir / f"{base}_metadata.json"
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, bundle_path)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return bundle_path, meta_path


def load_training_bundle(path: Path | str) -> tuple[Pipeline, dict[str, Any]]:
    """Load ``(pipeline, metadata)`` from a bundle written by :func:`save_training_bundle`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Bundle not found: {path}")
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj or "metadata" not in obj:
        raise ValueError(f"Unexpected bundle format: {path}")
    pipeline = obj["pipeline"]
    metadata = obj["metadata"]
    if not isinstance(pipeline, Pipeline):
        raise TypeError("Loaded object is not a sklearn Pipeline")
    return pipeline, metadata


def load_metadata_json(path: Path | str) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)
