#!/usr/bin/env python3
"""
Load demo transactions from JSON into PostgreSQL (transactions + optional model_predictions).

Run from the **repository root** so ``.env`` and ``DATABASE_URL`` resolve (same as the API).

Usage::

    python scripts/seed_postgres.py
    python scripts/seed_postgres.py --file data/demo/transactions.json --skip-existing

Expected console output (example): lines per inserted row, then a summary like
``Done. Inserted 19, skipped 5, errors 0.``
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.model_prediction import ModelPrediction
from app.models.transaction import Transaction

DEMO_MODEL_VERSION = "demo-seed"


def _parse_dt(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"transacted_at must be timezone-aware: {value!r}")
    return dt


def _load_rows(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("transactions")
    if not isinstance(rows, list):
        raise SystemExit("JSON must contain a top-level 'transactions' array")
    return rows


def _insert_transaction(db: Session, rec: dict) -> None:
    pred = rec.get("prediction") or {}
    label = (pred.get("prediction_label") or "legit").lower()
    score = float(pred.get("prediction_score", 0.0))
    is_fraud = label == "fraud"

    txn = Transaction(
        transaction_id=rec["transaction_id"],
        customer_id=rec["customer_id"],
        amount=Decimal(str(rec["amount"])),
        merchant=rec.get("merchant"),
        category=rec.get("category"),
        country=rec.get("country"),
        city=rec.get("city"),
        device_type=rec.get("device_type"),
        ip_address=rec.get("ip_address"),
        transacted_at=_parse_dt(rec["transacted_at"]),
        is_fraud_predicted=is_fraud,
        fraud_score=score,
        model_version=rec.get("model_version") or DEMO_MODEL_VERSION,
    )
    db.add(txn)
    db.flush()

    if pred:
        mp = ModelPrediction(
            transaction_id=txn.id,
            prediction_label=str(pred.get("prediction_label", "legit")),
            prediction_score=float(pred.get("prediction_score", 0.0)),
            top_risk_factors=pred.get("top_risk_factors"),
            latency_ms=pred.get("latency_ms"),
        )
        db.add(mp)


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed PostgreSQL from demo transaction JSON.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/demo/transactions.json"),
        help="Input JSON (default: data/demo/transactions.json)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows whose transaction_id already exists (default: exit with error on duplicate)",
    )
    args = parser.parse_args()
    if not args.file.is_file():
        raise SystemExit(f"File not found: {args.file.resolve()}")

    rows = _load_rows(args.file)
    inserted = 0
    skipped = 0
    errors = 0

    db = SessionLocal()
    try:
        for rec in rows:
            tid = rec["transaction_id"]
            existing = db.scalar(select(Transaction).where(Transaction.transaction_id == tid))
            if existing is not None:
                if args.skip_existing:
                    skipped += 1
                    print(f"skip  {tid} (already in database)")
                    continue
                raise SystemExit(f"Duplicate transaction_id: {tid!r} (use --skip-existing)")
            try:
                _insert_transaction(db, rec)
                db.commit()
                inserted += 1
                print(f"insert {tid}")
            except Exception as exc:  # noqa: BLE001 — CLI: show row context
                db.rollback()
                errors += 1
                print(f"error {tid}: {exc}")
    finally:
        db.close()

    print(f"Done. Inserted {inserted}, skipped {skipped}, errors {errors}.")


if __name__ == "__main__":
    main()
