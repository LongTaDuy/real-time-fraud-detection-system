from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, and_, func, select
from sqlalchemy.orm import Session, aliased

from app.models.model_prediction import ModelPrediction
from app.models.transaction import Transaction


@dataclass(frozen=True, slots=True)
class TransactionListFilters:
    customer_id: str | None
    merchant: str | None
    country: str | None
    predicted_label: str | None


class TransactionReadService:
    """Read-side queries for transactions and related predictions."""

    @staticmethod
    def _latest_prediction_per_transaction_subquery() -> Any:
        return (
            select(
                ModelPrediction.transaction_id.label("tid"),
                func.max(ModelPrediction.created_at).label("mx"),
            )
            .group_by(ModelPrediction.transaction_id)
            .subquery()
        )

    @staticmethod
    def _join_transaction_to_latest_prediction(
        stmt: Any,
        latest_sq: Any,
        mp: Any,
    ) -> Any:
        return stmt.join(latest_sq, Transaction.id == latest_sq.c.tid).join(
            mp,
            and_(
                mp.transaction_id == latest_sq.c.tid,
                mp.created_at == latest_sq.c.mx,
            ),
        )

    def _base_transaction_predicates(self, filters: TransactionListFilters) -> list[ColumnElement[bool]]:
        parts: list[ColumnElement[bool]] = []
        if filters.customer_id is not None:
            parts.append(Transaction.customer_id == filters.customer_id)
        if filters.merchant is not None:
            parts.append(Transaction.merchant == filters.merchant)
        if filters.country is not None:
            parts.append(Transaction.country == filters.country)
        return parts

    def list_transactions(
        self,
        db: Session,
        *,
        filters: TransactionListFilters,
        limit: int,
        offset: int,
    ) -> tuple[list[tuple[Transaction, str | None]], int]:
        base_where = self._base_transaction_predicates(filters)

        if filters.predicted_label is None:
            latest_label = (
                select(ModelPrediction.prediction_label)
                .where(ModelPrediction.transaction_id == Transaction.id)
                .order_by(ModelPrediction.created_at.desc())
                .limit(1)
                .scalar_subquery()
            )
            stmt = select(Transaction, latest_label.label("latest_prediction_label"))
            if base_where:
                stmt = stmt.where(*base_where)
            stmt = stmt.order_by(Transaction.created_at.desc()).limit(limit).offset(offset)

            count_stmt = select(func.count()).select_from(Transaction)
            if base_where:
                count_stmt = count_stmt.where(*base_where)
            total = int(db.scalar(count_stmt) or 0)

            rows = db.execute(stmt).all()
            return [(row[0], row[1]) for row in rows], total

        latest_sq = self._latest_prediction_per_transaction_subquery()
        mp = aliased(ModelPrediction)

        stmt = select(Transaction, mp.prediction_label.label("latest_prediction_label"))
        stmt = self._join_transaction_to_latest_prediction(stmt, latest_sq, mp)
        stmt = stmt.where(mp.prediction_label == filters.predicted_label)
        if base_where:
            stmt = stmt.where(*base_where)
        stmt = stmt.order_by(Transaction.created_at.desc()).limit(limit).offset(offset)

        count_stmt = select(func.count()).select_from(Transaction)
        count_stmt = self._join_transaction_to_latest_prediction(count_stmt, latest_sq, mp)
        count_stmt = count_stmt.where(mp.prediction_label == filters.predicted_label)
        if base_where:
            count_stmt = count_stmt.where(*base_where)

        total = int(db.scalar(count_stmt) or 0)
        rows = db.execute(stmt).all()
        return [(row[0], row[1]) for row in rows], total

    def get_by_business_id(
        self,
        db: Session,
        business_transaction_id: str,
    ) -> tuple[Transaction, list[ModelPrediction]] | None:
        tx = db.scalars(
            select(Transaction).where(Transaction.transaction_id == business_transaction_id)
        ).first()
        if tx is None:
            return None
        preds = db.scalars(
            select(ModelPrediction)
            .where(ModelPrediction.transaction_id == tx.id)
            .order_by(ModelPrediction.created_at.desc())
        ).all()
        return tx, list(preds)

    def list_recent_predictions(
        self,
        db: Session,
        *,
        limit: int,
        offset: int,
    ) -> tuple[list[tuple[ModelPrediction, Transaction]], int]:
        stmt = (
            select(ModelPrediction, Transaction)
            .join(Transaction, ModelPrediction.transaction_id == Transaction.id)
            .order_by(ModelPrediction.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        total = int(db.scalar(select(func.count()).select_from(ModelPrediction)) or 0)
        rows = db.execute(stmt).all()
        return [(row[0], row[1]) for row in rows], total
