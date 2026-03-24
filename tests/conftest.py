"""Shared fixtures: SQLite test DB (shared in-memory), fake Redis, patched app session."""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Register before any create_all — maps Postgres types to SQLite.
@compiles(JSONB, "sqlite")
def _compile_jsonb_sqlite(_element, compiler, **kw):  # type: ignore[no-untyped-def]
    return "JSON"


@compiles(UUID, "sqlite")
def _compile_uuid_sqlite(_element, compiler, **kw):  # type: ignore[no-untyped-def]
    return "CHAR(36)"


def pytest_configure(config: pytest.Config) -> None:
    """Ensure settings exist before ``app`` modules first import (SQLite + Redis URL placeholder)."""
    os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///:memory:")
    os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/15")
    os.environ.setdefault("FRAUD_CACHE_TTL_SECONDS", "300")
    os.environ.setdefault("API_RATE_LIMIT_PER_MINUTE", "0")
    os.environ.setdefault("FRAUD_POST_RATE_LIMIT_PER_MINUTE", "0")


@pytest.fixture
def redis_client():
    import fakeredis

    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def sqlite_engine() -> Generator:
    from app.db.base import Base
    from app.models.model_prediction import ModelPrediction  # noqa: F401
    from app.models.transaction import Transaction  # noqa: F401

    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def db_session(sqlite_engine) -> Generator[Session, None, None]:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sqlite_engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def app_client(sqlite_engine, redis_client, monkeypatch) -> Generator[TestClient, None, None]:
    import app.db.session as session_mod
    import app.redis_client as redis_mod
    from app.config import get_settings
    from app.main import create_app

    monkeypatch.delenv("MODEL_BUNDLE_PATH", raising=False)
    get_settings.cache_clear()
    factory = sessionmaker(autocommit=False, autoflush=False, bind=sqlite_engine)
    monkeypatch.setattr(session_mod, "engine", sqlite_engine)
    monkeypatch.setattr(session_mod, "SessionLocal", factory)
    monkeypatch.setattr(redis_mod, "_client", None)

    def _get_redis():
        return redis_client

    # ``from app.redis_client import get_redis`` in deps keeps a global name; patch it so
    # Depends(get_redis_client) resolves to the fake client.
    import app.api.deps as api_deps

    monkeypatch.setattr(redis_mod, "get_redis", _get_redis)
    monkeypatch.setattr(api_deps, "get_redis", _get_redis)

    app = create_app()
    with TestClient(app) as client:
        yield client


@pytest.fixture
def app_client_predictor_cleared(app_client: TestClient):
    """Temporarily clear the loaded predictor to assert 503 behavior; restores after the test."""
    from app.config import get_settings
    from app.ml.inference import FraudPredictor

    saved = app_client.app.state.fraud_predictor
    app_client.app.state.fraud_predictor = None
    yield app_client
    app_client.app.state.fraud_predictor = (
        saved if saved is not None else FraudPredictor.load(get_settings())
    )


@pytest.fixture
def score_request_body() -> dict:
    return {
        "transaction_id": "txn_test_001",
        "customer_id": "cus_test_a",
        "amount": "12.34",
        "merchant": "Acme Coffee",
        "category": "food_and_drink",
        "country": "US",
        "city": "Seattle",
        "device_type": "mobile",
        "ip_address": "192.0.2.10",
        "transacted_at": "2025-03-20T12:00:00+00:00",
    }


@pytest.fixture
def seeded_transaction(db_session: Session) -> tuple[uuid.UUID, str]:
    """Row in DB without Redis cache — used to assert 409 when reusing that id with a different payload."""
    from app.models.transaction import Transaction

    tid = "txn_seeded_conflict"
    tx = Transaction(
        transaction_id=tid,
        customer_id="cus_x",
        amount=Decimal("50.00"),
        merchant="Existing",
        category="retail",
        country="US",
        city="Boston",
        device_type="web",
        ip_address="192.0.2.20",
        transacted_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    db_session.add(tx)
    db_session.commit()
    db_session.refresh(tx)
    return tx.id, tid
