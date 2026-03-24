from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import Settings, get_settings
from app.ml.artifacts import save_training_bundle
from app.ml.inference import FraudPredictor


@pytest.fixture(autouse=True)
def _clear_settings_cache_after_inference_case() -> None:
    yield
    get_settings.cache_clear()


def test_load_without_bundle_path_uses_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_BUNDLE_PATH", raising=False)
    get_settings.cache_clear()
    settings = Settings(
        database_url="sqlite+pysqlite:///:memory:",
        redis_url="redis://localhost:6379/0",
        model_bundle_path=None,
    )
    p = FraudPredictor.load(settings)
    assert p.is_stub is True


def test_load_missing_file_falls_back_to_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    missing = tmp_path / "nope_bundle.joblib"
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(missing))
    get_settings.cache_clear()
    settings = get_settings()
    p = FraudPredictor.load(settings)
    assert p.is_stub is True


def test_load_corrupt_bundle_falls_back_to_stub(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad = tmp_path / "bad_bundle.joblib"
    bad.write_bytes(b"not a real joblib payload")
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bad))
    get_settings.cache_clear()
    p = FraudPredictor.load(get_settings())
    assert p.is_stub is True


def test_load_valid_bundle_runs_non_stub_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rng = np.random.RandomState(0)
    X = rng.randn(40, 2)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(int)
    pipe = Pipeline(
        [
            ("preprocess", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    metadata = {
        "feature_columns_numeric": ["V1", "V2"],
        "feature_columns_categorical": [],
        "positive_label": 1,
        "model_version": "pytest-bundle",
    }
    bundle_path, _ = save_training_bundle(
        tmp_path,
        pipe,
        metadata,
        model_slug="pytest",
        model_version="0.0.1",
    )
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    get_settings.cache_clear()
    p = FraudPredictor.load(get_settings())
    assert p.is_stub is False

    from datetime import datetime, timezone
    from decimal import Decimal

    from app.schemas.fraud import FraudScoreRequest

    req = FraudScoreRequest(
        transaction_id="txn_ml_1",
        customer_id="cus_ml",
        amount=Decimal("10.00"),
        merchant="M",
        category="c",
        country="US",
        city="X",
        device_type="web",
        ip_address="192.0.2.1",
        transacted_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        model_features={"V1": 0.5, "V2": -0.25},
    )
    out = p.predict(req, model_features=req.model_features)
    assert out.model_version == "pytest-bundle"
    assert out.prediction_label in ("fraud", "legit")
    assert 0.0 <= out.fraud_score <= 1.0


def test_non_stub_missing_required_features_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rng = np.random.RandomState(1)
    X = rng.randn(30, 1)
    y = (X[:, 0] > 0).astype(int)
    pipe = Pipeline(
        [
            ("preprocess", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=500, random_state=0)),
        ]
    )
    pipe.fit(X, y)
    metadata = {
        "feature_columns_numeric": ["V1", "V2"],
        "feature_columns_categorical": [],
        "positive_label": 1,
        "model_version": "pytest-missing",
    }
    bundle_path, _ = save_training_bundle(
        tmp_path,
        pipe,
        metadata,
        model_slug="pytest2",
        model_version="0.0.2",
    )
    monkeypatch.setenv("MODEL_BUNDLE_PATH", str(bundle_path))
    get_settings.cache_clear()
    p = FraudPredictor.load(get_settings())
    assert p.is_stub is False

    from datetime import datetime, timezone
    from decimal import Decimal

    from app.schemas.fraud import FraudScoreRequest

    req = FraudScoreRequest(
        transaction_id="txn_ml_2",
        customer_id="cus_ml",
        amount=Decimal("10.00"),
        transacted_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        model_features={"V1": 1.0},
    )
    with pytest.raises(ValueError, match="V2"):
        p.predict(req, model_features=req.model_features)
