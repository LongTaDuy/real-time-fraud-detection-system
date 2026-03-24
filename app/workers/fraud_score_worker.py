"""Background worker: consume fraud score jobs from Redis and persist via FraudScoringService.

Run::

    python -m app.workers.fraud_score_worker

Or with Docker Compose service ``worker`` (see docker-compose.yml).
"""

from __future__ import annotations

import signal

from app.config import get_settings
from app.db.session import SessionLocal
from app.observability import configure_observability, get_logger
from app.ml.inference import FraudPredictor
from app.queue.redis_score_queue import FraudScoreJobQueue
from app.redis_client import build_redis_client
from app.schemas.fraud import FraudScoreRequest
from app.services.cache_service import CacheService
from app.services.errors import (
    DuplicateTransactionError,
    FraudModelInputError,
    TransactionPayloadMismatchError,
)
from app.services.fraud_scoring import FraudScoringService
from app.services.transaction_persistence import TransactionPersistenceService

log = get_logger(__name__)

_shutdown = False


def _handle_sigterm(_signum: int, _frame: object) -> None:
    global _shutdown
    _shutdown = True
    log.info("worker_shutdown_requested")


def run() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    settings = get_settings()
    configure_observability(settings)
    predictor = FraudPredictor.load(settings)
    redis_client = build_redis_client(
        redis_url=settings.redis_url,
        socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )
    queue = FraudScoreJobQueue(redis_client, settings.fraud_score_queue_key)
    service = FraudScoringService(
        predictor=predictor,
        cache=CacheService(redis_client),
        persistence=TransactionPersistenceService(),
        cache_ttl_seconds=settings.fraud_cache_ttl_seconds,
    )

    log.info(
        "worker_started",
        queue_key=queue.key,
        predictor_stub=predictor.is_stub,
        model_version=predictor.reportable_model_version,
    )

    brpop_timeout = 5
    while not _shutdown:
        raw = queue.blocking_pop_raw(timeout_seconds=brpop_timeout)
        if raw is None:
            continue
        try:
            body = FraudScoreRequest.model_validate_json(raw)
        except Exception:
            log.exception("worker_job_invalid_payload", detail="invalid JSON or schema")
            continue

        db = SessionLocal()
        try:
            service.score_transaction(db, body)
            log.info("worker_job_scored", transaction_id=body.transaction_id)
        except DuplicateTransactionError as exc:
            log.info("worker_job_skipped_duplicate", transaction_id=exc.transaction_id)
        except TransactionPayloadMismatchError as exc:
            log.warning(
                "worker_job_payload_mismatch",
                transaction_id=exc.transaction_id,
                detail=str(exc),
            )
        except FraudModelInputError as exc:
            log.warning(
                "worker_job_model_input_error",
                transaction_id=body.transaction_id,
                error=str(exc),
            )
        except Exception:
            log.exception("worker_job_scoring_failed", transaction_id=body.transaction_id)
            db.rollback()
        finally:
            db.close()

    redis_client.close()
    log.info("worker_exited")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
