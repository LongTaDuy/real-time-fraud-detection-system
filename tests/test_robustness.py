from __future__ import annotations

from fastapi.testclient import TestClient


def test_service_unavailable_when_predictor_missing(
    app_client_predictor_cleared: TestClient,
    score_request_body: dict,
) -> None:
    r = app_client_predictor_cleared.post("/api/v1/fraud/score", json=score_request_body)
    assert r.status_code == 503
    body = r.json()
    assert "detail" in body
    combined = str(body).lower()
    assert "unavailable" in combined


def test_content_length_too_large_returns_413(app_client: TestClient, score_request_body: dict) -> None:
    import json

    raw = json.dumps(score_request_body).encode("utf-8")
    r = app_client.post(
        "/api/v1/fraud/score",
        content=raw,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(10_000_000),
        },
    )
    assert r.status_code == 413


def test_customer_id_control_char_rejected(app_client: TestClient, score_request_body: dict) -> None:
    body = {**score_request_body, "customer_id": "bad\nid", "transaction_id": "txn_ctrl_1"}
    r = app_client.post("/api/v1/fraud/score", json=body)
    assert r.status_code == 422


def test_model_features_bad_key_rejected(app_client: TestClient, score_request_body: dict) -> None:
    body = {
        **score_request_body,
        "transaction_id": "txn_bad_mf",
        "model_features": {"bad key!": 1.0},
    }
    r = app_client.post("/api/v1/fraud/score", json=body)
    assert r.status_code == 422
