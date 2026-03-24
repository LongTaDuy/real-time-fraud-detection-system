from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_fraud_score_queue, get_fraud_scoring_service
from app.api.http_exceptions import (
    duplicate_transaction_conflict,
    fraud_model_input_unprocessable,
    transaction_payload_mismatch_conflict,
)
from app.queue.redis_score_queue import FraudScoreJobQueue
from app.schemas.fraud import (
    FraudScoreApiResponse,
    FraudScoreAsyncAcceptedResponse,
    FraudScoreRequest,
)
from app.observability import get_logger
from app.services.errors import (
    DuplicateTransactionError,
    FraudModelInputError,
    TransactionPayloadMismatchError,
)
from app.services.fraud_scoring import FraudScoringService

router = APIRouter(prefix="/api/v1/fraud", tags=["fraud"])
log = get_logger(__name__)


@router.post(
    "/score",
    response_model=FraudScoreApiResponse,
    status_code=status.HTTP_200_OK,
)
def score_transaction(
    body: FraudScoreRequest,
    db: Session = Depends(get_db),
    fraud_service: FraudScoringService = Depends(get_fraud_scoring_service),
) -> FraudScoreApiResponse:
    try:
        result = fraud_service.score_transaction(db, body)
    except DuplicateTransactionError as exc:
        log.warning(
            "fraud_score_rejected",
            reason="duplicate_transaction",
            transaction_id=body.transaction_id,
        )
        raise duplicate_transaction_conflict(exc) from exc
    except TransactionPayloadMismatchError as exc:
        log.warning(
            "fraud_score_rejected",
            reason="transaction_payload_mismatch",
            transaction_id=body.transaction_id,
        )
        raise transaction_payload_mismatch_conflict(exc) from exc
    except FraudModelInputError as exc:
        log.warning(
            "fraud_score_rejected",
            reason="model_input_error",
            transaction_id=body.transaction_id,
            error=str(exc),
        )
        raise fraud_model_input_unprocessable(exc) from exc
    return FraudScoreApiResponse.from_domain(result)


@router.post(
    "/score-async",
    response_model=FraudScoreAsyncAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue fraud scoring (async)",
)
def score_transaction_async(
    body: FraudScoreRequest,
    queue: FraudScoreJobQueue = Depends(get_fraud_score_queue),
) -> FraudScoreAsyncAcceptedResponse:
    queue.enqueue(body)
    log.info(
        "fraud_score_async_enqueued",
        transaction_id=body.transaction_id,
    )
    return FraudScoreAsyncAcceptedResponse(transaction_id=body.transaction_id)
