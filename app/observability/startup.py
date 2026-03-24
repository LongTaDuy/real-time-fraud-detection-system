"""Startup connectivity probes (log-only; app still serves /health/ready for truth)."""

from __future__ import annotations

import structlog
from sqlalchemy import text

from app.config import Settings
from app.db.session import engine
from app.redis_client import build_redis_client

log = structlog.get_logger("app.startup")


def log_startup_connectivity(settings: Settings) -> None:
    """Log Postgres and Redis reachability once at boot (errors do not stop the process)."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log.info("postgres_connectivity_ok", detail="SELECT 1 succeeded")
    except Exception:
        log.exception("postgres_connectivity_failed", detail="engine.connect or SELECT 1 failed")

    redis_client = None
    try:
        redis_client = build_redis_client(
            redis_url=settings.redis_url,
            socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
        )
        redis_client.ping()
        log.info("redis_connectivity_ok", detail="PING succeeded")
    except Exception:
        log.exception("redis_connectivity_failed", detail="Redis PING failed")
    finally:
        if redis_client is not None:
            redis_client.close()
