from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from app.models.model_prediction import ModelPrediction
from app.models.transaction import Transaction
from app.services.transaction_read_service import TransactionListFilters, TransactionReadService


def _tx(**kwargs: object) -> Transaction:
    defaults = {
        "transaction_id": "t1",
        "customer_id": "c1",
        "amount": Decimal("10.00"),
        "merchant": "M1",
        "category": "retail",
        "country": "US",
        "city": "NYC",
        "device_type": "mobile",
        "ip_address": "192.0.2.1",
        "transacted_at": datetime(2025, 5, 1, tzinfo=timezone.utc),
    }
    defaults.update(kwargs)
    return Transaction(**defaults)  # type: ignore[arg-type]


def test_list_filters_by_customer_id(db_session) -> None:
    db_session.add(_tx(transaction_id="a1", customer_id="alice"))
    db_session.add(_tx(transaction_id="a2", customer_id="bob"))
    db_session.commit()

    svc = TransactionReadService()
    rows, total = svc.list_transactions(
        db_session,
        filters=TransactionListFilters(
            customer_id="alice",
            merchant=None,
            country=None,
            predicted_label=None,
        ),
        limit=10,
        offset=0,
    )
    assert total == 1
    assert rows[0][0].transaction_id == "a1"


def test_list_filters_by_latest_prediction_label(db_session) -> None:
    t_legit = _tx(transaction_id="legit_row", customer_id="c")
    t_fraud = _tx(transaction_id="fraud_row", customer_id="c")
    db_session.add_all([t_legit, t_fraud])
    db_session.flush()

    p_old = ModelPrediction(
        transaction_id=t_fraud.id,
        prediction_label="legit",
        prediction_score=0.2,
        top_risk_factors=None,
        latency_ms=1,
        created_at=datetime(2025, 5, 2, 10, 0, 0, tzinfo=timezone.utc),
    )
    p_new = ModelPrediction(
        transaction_id=t_fraud.id,
        prediction_label="fraud",
        prediction_score=0.9,
        top_risk_factors=None,
        latency_ms=1,
        created_at=datetime(2025, 5, 2, 11, 0, 0, tzinfo=timezone.utc),
    )
    db_session.add_all([p_old, p_new])
    db_session.commit()

    svc = TransactionReadService()
    rows, total = svc.list_transactions(
        db_session,
        filters=TransactionListFilters(
            customer_id=None,
            merchant=None,
            country=None,
            predicted_label="fraud",
        ),
        limit=10,
        offset=0,
    )
    assert total == 1
    assert rows[0][0].transaction_id == "fraud_row"


def test_get_by_business_id_returns_predictions_newest_first(db_session) -> None:
    t = _tx(transaction_id="detail_1")
    db_session.add(t)
    db_session.flush()
    db_session.add(
        ModelPrediction(
            transaction_id=t.id,
            prediction_label="legit",
            prediction_score=0.1,
            top_risk_factors=None,
            latency_ms=5,
            created_at=datetime(2025, 5, 3, 8, 0, 0, tzinfo=timezone.utc),
        )
    )
    db_session.add(
        ModelPrediction(
            transaction_id=t.id,
            prediction_label="fraud",
            prediction_score=0.99,
            top_risk_factors=None,
            latency_ms=6,
            created_at=datetime(2025, 5, 3, 9, 0, 0, tzinfo=timezone.utc),
        )
    )
    db_session.commit()

    svc = TransactionReadService()
    out = svc.get_by_business_id(db_session, "detail_1")
    assert out is not None
    tx, preds = out
    assert tx.transaction_id == "detail_1"
    assert [p.prediction_label for p in preds] == ["fraud", "legit"]
