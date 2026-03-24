from __future__ import annotations

from fastapi.testclient import TestClient


def test_score_rejects_non_positive_amount(app_client: TestClient, score_request_body: dict) -> None:
    body = {**score_request_body, "amount": "0.00", "transaction_id": "txn_bad_amt"}
    r = app_client.post("/api/v1/fraud/score", json=body)
    assert r.status_code == 422


def test_score_rejects_invalid_transaction_id_chars(
    app_client: TestClient,
    score_request_body: dict,
) -> None:
    body = {**score_request_body, "transaction_id": "bad id!"}
    r = app_client.post("/api/v1/fraud/score", json=body)
    assert r.status_code == 422


def test_score_rejects_invalid_ip(app_client: TestClient, score_request_body: dict) -> None:
    body = {**score_request_body, "ip_address": "999.999.999.999", "transaction_id": "txn_bad_ip"}
    r = app_client.post("/api/v1/fraud/score", json=body)
    assert r.status_code == 422


def test_score_rejects_unknown_fields(app_client: TestClient, score_request_body: dict) -> None:
    body = {**score_request_body, "extra_field": 1, "transaction_id": "txn_extra"}
    r = app_client.post("/api/v1/fraud/score", json=body)
    assert r.status_code == 422


def test_list_rejects_invalid_predicted_label(app_client: TestClient) -> None:
    r = app_client.get("/api/v1/transactions", params={"predicted_label": "not@valid"})
    assert r.status_code == 422


def test_list_rejects_country_too_long(app_client: TestClient) -> None:
    r = app_client.get("/api/v1/transactions", params={"country": "TOOLONGCODE"})
    assert r.status_code == 422
