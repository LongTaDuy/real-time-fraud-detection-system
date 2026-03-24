from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.schemas.fraud import FraudScoreRequest
from app.services.errors import DuplicateTransactionError
from app.services.transaction_persistence import TransactionPersistenceService


@pytest.fixture
def sample_fraud_request() -> FraudScoreRequest:
    return FraudScoreRequest(
        transaction_id="txn_persist_1",
        customer_id="cus_persist",
        amount=Decimal("44.00"),
        merchant="Shop",
        category="retail",
        country="US",
        city="Denver",
        device_type="pos",
        ip_address="192.0.2.30",
        transacted_at=datetime(2025, 4, 1, 15, 30, tzinfo=timezone.utc),
    )


def test_create_transaction_round_trip(db_session, sample_fraud_request: FraudScoreRequest) -> None:
    svc = TransactionPersistenceService()
    row = svc.create_transaction(db_session, sample_fraud_request)
    db_session.commit()
    assert row.id is not None
    assert row.transaction_id == sample_fraud_request.transaction_id
    assert row.amount == sample_fraud_request.amount

    found = svc.get_by_business_id(db_session, sample_fraud_request.transaction_id)
    assert found is not None
    assert found.id == row.id


def test_duplicate_business_transaction_id_raises(
    db_session,
    sample_fraud_request: FraudScoreRequest,
) -> None:
    svc = TransactionPersistenceService()
    svc.create_transaction(db_session, sample_fraud_request)
    db_session.commit()

    with pytest.raises(DuplicateTransactionError):
        svc.create_transaction(db_session, sample_fraud_request)
