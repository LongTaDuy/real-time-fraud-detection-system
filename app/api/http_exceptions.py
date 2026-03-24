"""Map domain errors from the fraud stack to FastAPI ``HTTPException`` instances."""

from __future__ import annotations

from fastapi import HTTPException, status

from app.services.errors import (
    DuplicateTransactionError,
    FraudModelInputError,
    TransactionPayloadMismatchError,
)


def duplicate_transaction_conflict(exc: DuplicateTransactionError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={"message": str(exc), "transaction_id": exc.transaction_id},
    )


def fraud_model_input_unprocessable(exc: FraudModelInputError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail={"message": str(exc)},
    )


def transaction_payload_mismatch_conflict(exc: TransactionPayloadMismatchError) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail={
            "message": str(exc),
            "transaction_id": exc.transaction_id,
            "reason": "transaction_id_reuse_with_different_payload",
        },
    )
