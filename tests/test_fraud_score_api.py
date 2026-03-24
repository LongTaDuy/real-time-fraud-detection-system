from __future__ import annotations

from fastapi.testclient import TestClient


def test_score_transaction_sync_success(app_client: TestClient, score_request_body: dict) -> None:
    r = app_client.post("/api/v1/fraud/score", json=score_request_body)
    assert r.status_code == 200
    data = r.json()
    assert data["transaction_id"] == score_request_body["transaction_id"]
    assert data["cached"] is False
    assert "fraud_score" in data
    assert 0.0 <= data["fraud_score"] <= 1.0
    assert data["predicted_label"] in ("fraud", "legit")
    assert data["latency_ms"] >= 0
    assert data["model_version"]


def test_score_second_request_is_cache_hit(app_client: TestClient, score_request_body: dict) -> None:
    first = app_client.post("/api/v1/fraud/score", json=score_request_body)
    assert first.status_code == 200
    assert first.json()["cached"] is False

    second = app_client.post("/api/v1/fraud/score", json=score_request_body)
    assert second.status_code == 200
    body = second.json()
    assert body["cached"] is True
    assert body["transaction_id"] == score_request_body["transaction_id"]
    assert body["fraud_score"] == first.json()["fraud_score"]


def test_score_duplicate_transaction_returns_409(
    app_client: TestClient,
    score_request_body: dict,
    seeded_transaction: tuple,
) -> None:
    _, business_id = seeded_transaction
    body = {**score_request_body, "transaction_id": business_id}
    r = app_client.post("/api/v1/fraud/score", json=body)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["transaction_id"] == business_id
    assert "message" in detail
    assert detail.get("reason") == "transaction_id_reuse_with_different_payload"


def test_score_same_transaction_id_different_payload_returns_409(
    app_client: TestClient,
    score_request_body: dict,
) -> None:
    first = app_client.post("/api/v1/fraud/score", json=score_request_body)
    assert first.status_code == 200

    conflict = {
        **score_request_body,
        "amount": "99.99",
    }
    r = app_client.post("/api/v1/fraud/score", json=conflict)
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["transaction_id"] == score_request_body["transaction_id"]
    assert detail.get("reason") == "transaction_id_reuse_with_different_payload"


def test_score_async_accepted(app_client: TestClient, score_request_body: dict) -> None:
    body = {**score_request_body, "transaction_id": "txn_async_only_1"}
    r = app_client.post("/api/v1/fraud/score-async", json=body)
    assert r.status_code == 202
    assert r.json()["status"] == "accepted"
    assert r.json()["transaction_id"] == "txn_async_only_1"
