"""initial schema: transactions and model_predictions

Revision ID: 20250324_0001
Revises:
Create Date: 2025-03-24

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20250324_0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_id", sa.String(length=128), nullable=False),
        sa.Column("customer_id", sa.String(length=128), nullable=False),
        sa.Column("amount", sa.Numeric(precision=14, scale=2), nullable=False),
        sa.Column("merchant", sa.String(length=255), nullable=True),
        sa.Column("category", sa.String(length=128), nullable=True),
        sa.Column("country", sa.String(length=8), nullable=True),
        sa.Column("city", sa.String(length=128), nullable=True),
        sa.Column("device_type", sa.String(length=64), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("is_fraud_predicted", sa.Boolean(), nullable=True),
        sa.Column("fraud_score", sa.Float(), nullable=True),
        sa.Column("model_version", sa.String(length=64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("transaction_id", name="uq_transactions_transaction_id"),
    )
    op.create_index(
        "ix_transactions_customer_id",
        "transactions",
        ["customer_id"],
        unique=False,
    )
    op.create_index(
        "ix_transactions_timestamp",
        "transactions",
        ["timestamp"],
        unique=False,
    )
    op.create_index(
        "ix_transactions_created_at",
        "transactions",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "ix_transactions_is_fraud_predicted",
        "transactions",
        ["is_fraud_predicted"],
        unique=False,
    )

    op.create_table(
        "model_predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("transaction_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("prediction_label", sa.String(length=32), nullable=False),
        sa.Column("prediction_score", sa.Float(), nullable=False),
        sa.Column("top_risk_factors", postgresql.JSONB(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["transaction_id"],
            ["transactions.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_model_predictions_transaction_id",
        "model_predictions",
        ["transaction_id"],
        unique=False,
    )
    op.create_index(
        "ix_model_predictions_created_at",
        "model_predictions",
        ["created_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_model_predictions_created_at",
        table_name="model_predictions",
    )
    op.drop_index(
        "ix_model_predictions_transaction_id",
        table_name="model_predictions",
    )
    op.drop_table("model_predictions")
    op.drop_index(
        "ix_transactions_is_fraud_predicted",
        table_name="transactions",
    )
    op.drop_index("ix_transactions_created_at", table_name="transactions")
    op.drop_index("ix_transactions_timestamp", table_name="transactions")
    op.drop_index("ix_transactions_customer_id", table_name="transactions")
    op.drop_table("transactions")
