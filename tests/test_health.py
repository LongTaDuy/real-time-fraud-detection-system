from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(app_client: TestClient) -> None:
    r = app_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


def test_ready_probes_database_and_redis(app_client: TestClient) -> None:
    r = app_client.get("/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["database"] == "connected"
    assert body["redis"] == "connected"
