from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_transaction_read_service
from app.api.filter_params import optional_trimmed, transaction_list_filters
from app.schemas.pagination import PaginationMeta
from app.schemas.prediction import PredictionRecord
from app.schemas.transaction import TransactionResponse
from app.schemas.transaction_read import (
    TransactionDetailResponse,
    TransactionListResponse,
    TransactionSummary,
)
from app.services.transaction_read_service import TransactionReadService

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])


@router.get(
    "",
    response_model=TransactionListResponse,
    summary="List transactions",
)
def list_transactions(
    db: Session = Depends(get_db),
    service: TransactionReadService = Depends(get_transaction_read_service),
    limit: int = Query(20, ge=1, le=100, description="Page size"),
    offset: int = Query(0, ge=0, description="Rows to skip"),
    customer_id: str | None = Query(None, description="Exact match on customer_id"),
    merchant: str | None = Query(None, description="Exact match on merchant"),
    country: str | None = Query(None, description="Exact match on country"),
    predicted_label: str | None = Query(
        None,
        description="Filter to transactions whose *latest* prediction has this label",
    ),
) -> TransactionListResponse:
    filters = transaction_list_filters(
        customer_id=customer_id,
        merchant=merchant,
        country=country,
        predicted_label=predicted_label,
    )
    rows, total = service.list_transactions(db, filters=filters, limit=limit, offset=offset)
    items = [TransactionSummary.from_list_row(tx, latest_label) for tx, latest_label in rows]
    return TransactionListResponse(
        items=items,
        pagination=PaginationMeta(limit=limit, offset=offset, total=total),
    )


@router.get(
    "/{transaction_id}",
    response_model=TransactionDetailResponse,
    summary="Get one transaction by business transaction_id",
)
def get_transaction(
    transaction_id: str = Path(
        ...,
        min_length=1,
        max_length=128,
        description="External transaction_id (unique business key)",
    ),
    db: Session = Depends(get_db),
    service: TransactionReadService = Depends(get_transaction_read_service),
) -> TransactionDetailResponse:
    tid = optional_trimmed(transaction_id, max_length=128, field="transaction_id")
    if tid is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"message": "transaction_id must not be blank"},
        )
    result = service.get_by_business_id(db, tid)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"message": "transaction not found", "transaction_id": tid},
        )
    tx, preds = result
    return TransactionDetailResponse(
        transaction=TransactionResponse.from_transaction(tx),
        predictions=[PredictionRecord.from_prediction(p) for p in preds],
    )
