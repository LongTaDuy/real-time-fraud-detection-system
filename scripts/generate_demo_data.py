#!/usr/bin/env python3
"""
Generate realistic fintech-style demo transactions (JSON).

Usage (from repo root)::

    python scripts/generate_demo_data.py
    python scripts/generate_demo_data.py --output data/demo/transactions.json --count 24

Expected console output: one line like
``Wrote 24 transactions (5 tagged suspicious) -> <absolute path>``
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

MERCHANTS = [
    ("Starbucks", "food_and_drink"),
    ("Whole Foods Market", "grocery"),
    ("Shell", "gas"),
    ("Uber", "transport"),
    ("Amazon", "retail"),
    ("Netflix", "subscription"),
    ("Delta Air Lines", "travel"),
    ("CVS Pharmacy", "health"),
    ("Spotify", "subscription"),
    ("Target", "retail"),
]

CITIES = [
    ("US", "Seattle"),
    ("US", "Austin"),
    ("US", "Chicago"),
    ("GB", "London"),
    ("CA", "Toronto"),
    ("DE", "Berlin"),
]

SUSPICIOUS_PRESETS: list[dict] = [
    {
        "transaction_id": "txn_demo_susp_wire_round",
        "customer_id": "cus_demo_new_7fz",
        "amount": "9500.00",
        "merchant": "Offshore Wire Services Ltd",
        "category": "wire_transfer",
        "country": "NG",
        "city": "Lagos",
        "device_type": "web",
        "ip_address": "102.89.12.44",
        "transacted_at": "2025-03-18T14:22:11+00:00",
        "tags": ["suspicious"],
        "prediction": {
            "prediction_label": "fraud",
            "prediction_score": 0.94,
            "latency_ms": 21,
            "top_risk_factors": {
                "method": "demo-seed",
                "headline": "Demo: large round wire + high-risk geography.",
                "factors": [
                    {
                        "feature": "amount",
                        "label": "Amount",
                        "direction": "increases_fraud_risk",
                        "strength": 1.0,
                        "contribution": 0.0,
                        "message": "Unusually large round amount for a new customer profile (demo).",
                    }
                ],
            },
        },
    },
    {
        "transaction_id": "txn_demo_susp_crypto",
        "customer_id": "cus_demo_risk_9k",
        "amount": "4200.50",
        "merchant": "QuickCrypto Exchange",
        "category": "crypto",
        "country": "US",
        "city": "Miami",
        "device_type": "web",
        "ip_address": "185.220.101.3",
        "transacted_at": "2025-03-18T15:05:33+00:00",
        "tags": ["suspicious"],
        "prediction": {
            "prediction_label": "fraud",
            "prediction_score": 0.88,
            "latency_ms": 18,
            "top_risk_factors": {
                "method": "demo-seed",
                "headline": "Demo: crypto cash-out pattern with datacenter-like IP.",
                "factors": [],
            },
        },
    },
    {
        "transaction_id": "txn_demo_susp_gambling",
        "customer_id": "cus_demo_std_2a",
        "amount": "2500.00",
        "merchant": "LuckySpin Online Casino",
        "category": "gambling",
        "country": "MT",
        "city": "Valletta",
        "device_type": "mobile",
        "ip_address": "89.45.12.8",
        "transacted_at": "2025-03-18T16:40:02+00:00",
        "tags": ["suspicious"],
        "prediction": {
            "prediction_label": "fraud",
            "prediction_score": 0.81,
            "latency_ms": 15,
            "top_risk_factors": {
                "method": "demo-seed",
                "headline": "Demo: gambling merchant + elevated ticket size.",
                "factors": [],
            },
        },
    },
    {
        "transaction_id": "txn_demo_susp_velocity",
        "customer_id": "cus_demo_std_2a",
        "amount": "899.99",
        "merchant": "Best Buy",
        "category": "electronics",
        "country": "US",
        "city": "Miami",
        "device_type": "mobile",
        "ip_address": "73.15.42.10",
        "transacted_at": "2025-03-18T16:41:18+00:00",
        "tags": ["suspicious"],
        "prediction": {
            "prediction_label": "fraud",
            "prediction_score": 0.76,
            "latency_ms": 14,
            "top_risk_factors": {
                "method": "demo-seed",
                "headline": "Demo: velocity / account takeover story (same customer_id).",
                "factors": [],
            },
        },
    },
    {
        "transaction_id": "txn_demo_susp_atm_abroad",
        "customer_id": "cus_demo_travel_4c",
        "amount": "800.00",
        "merchant": "INTL ATM WITHDRAWAL",
        "category": "cash_withdrawal",
        "country": "RU",
        "city": "Moscow",
        "device_type": "atm",
        "ip_address": "95.31.44.2",
        "transacted_at": "2025-03-19T03:12:45+00:00",
        "tags": ["suspicious"],
        "prediction": {
            "prediction_label": "fraud",
            "prediction_score": 0.79,
            "latency_ms": 17,
            "top_risk_factors": {
                "method": "demo-seed",
                "headline": "Demo: foreign ATM withdrawal inconsistent with typical home country.",
                "factors": [],
            },
        },
    },
]


def _random_baseline(rng: random.Random, base_time: datetime, i: int) -> dict:
    merchant, category = rng.choice(MERCHANTS)
    country, city = rng.choice(CITIES)
    amount = (Decimal(rng.randint(5, 250)) + Decimal(rng.randint(0, 99)) / 100).quantize(
        Decimal("0.01")
    )
    is_fraud = rng.random() < 0.12
    fraud_score = rng.uniform(0.55, 0.78) if is_fraud else rng.uniform(0.02, 0.38)
    label = "fraud" if is_fraud else "legit"
    ts = base_time + timedelta(minutes=i * 7 + rng.randint(0, 20))
    return {
        "transaction_id": f"txn_demo_gen_{i:04d}",
        "customer_id": f"cus_demo_{rng.randint(1000, 9999)}",
        "amount": format(amount, "f"),
        "merchant": merchant,
        "category": category,
        "country": country,
        "city": city,
        "device_type": rng.choice(["mobile", "web", "pos"]),
        "ip_address": f"{rng.randint(1,223)}.{rng.randint(0,255)}.{rng.randint(0,255)}.{rng.randint(1,254)}",
        "transacted_at": ts.replace(tzinfo=timezone.utc).isoformat(),
        "tags": ["baseline"],
        "prediction": {
            "prediction_label": label,
            "prediction_score": round(fraud_score, 4),
            "latency_ms": rng.randint(8, 25),
            "top_risk_factors": {
                "method": "demo-seed",
                "headline": "Demo seed — synthetic routine spend.",
                "factors": [],
            },
        },
    }


def build_dataset(*, count: int, seed: int) -> dict:
    rng = random.Random(seed)
    base_time = datetime(2025, 3, 18, 12, 0, 0, tzinfo=timezone.utc)
    n_presets = len(SUSPICIOUS_PRESETS)
    if count >= n_presets:
        n_baseline = count - n_presets
        suspicious = SUSPICIOUS_PRESETS
    else:
        n_baseline = 0
        suspicious = SUSPICIOUS_PRESETS[:count]
    txs = [_random_baseline(rng, base_time, i) for i in range(n_baseline)]
    txs.extend(json.loads(json.dumps(suspicious)))
    txs.sort(key=lambda r: r["transacted_at"])
    return {
        "meta": {
            "version": "1",
            "generator": "scripts/generate_demo_data.py",
            "seed": seed,
            "description": "Synthetic fintech transactions for demos, README screenshots, and interviews.",
        },
        "transactions": txs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate demo transaction JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/demo/transactions.json"),
        help="Output path (default: data/demo/transactions.json)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=24,
        help="Total rows (includes 5 built-in suspicious scenarios)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducible baselines")
    args = parser.parse_args()
    data = build_dataset(count=args.count, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    n = len(data["transactions"])
    n_sus = sum(1 for t in data["transactions"] if "suspicious" in (t.get("tags") or []))
    print(f"Wrote {n} transactions ({n_sus} tagged suspicious) -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
