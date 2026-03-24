from __future__ import annotations

from fastapi.testclient import TestClient


def test_list_transactions_empty(app_client: TestClient) -> None:
    r = app_client.get("/api/v1/transactions")
    assert r.status_code == 200
    data = r.json()
    assert data["items"] == []
    assert data["pagination"]["total"] == 0


def test_list_and_get_after_score(app_client: TestClient, score_request_body: dict) -> None:
    score_request_body = {**score_request_body, "transaction_id": "txn_list_flow_1"}
    assert app_client.post("/api/v1/fraud/score", json=score_request_body).status_code == 200

    listed = app_client.get("/api/v1/transactions")
    assert listed.status_code == 200
    items = listed.json()["items"]
    assert len(items) == 1
    assert items[0]["transaction_id"] == "txn_list_flow_1"
    assert items[0]["customer_id"] == score_request_body["customer_id"]

    one = app_client.get("/api/v1/transactions/txn_list_flow_1")
    assert one.status_code == 200
    detail = one.json()
    assert detail["transaction"]["transaction_id"] == "txn_list_flow_1"
    assert len(detail["predictions"]) >= 1


def test_get_transaction_not_found(app_client: TestClient) -> None:
    r = app_client.get("/api/v1/transactions/does-not-exist-123")
    assert r.status_code == 404
    assert r.json()["detail"]["transaction_id"] == "does-not-exist-123"
