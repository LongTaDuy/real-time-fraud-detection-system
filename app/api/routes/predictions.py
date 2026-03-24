from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_transaction_read_service
from app.schemas.pagination import PaginationMeta
from app.schemas.prediction import RecentPredictionItem, RecentPredictionsResponse
from app.services.transaction_read_service import TransactionReadService

router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


@router.get(
    "/recent",
    response_model=RecentPredictionsResponse,
    summary="Recent model predictions across all transactions",
)
def list_recent_predictions(
    db: Session = Depends(get_db),
    service: TransactionReadService = Depends(get_transaction_read_service),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> RecentPredictionsResponse:
    rows, total = service.list_recent_predictions(db, limit=limit, offset=offset)
    items = [
        RecentPredictionItem.from_prediction_and_transaction(pred, tx) for pred, tx in rows
    ]
    return RecentPredictionsResponse(
        items=items,
        pagination=PaginationMeta(limit=limit, offset=offset, total=total),
    )
