#!/usr/bin/env python3
"""
POST demo transactions to the fraud scoring API (sync or async).

Run from the **repository root**. Default base URL matches Docker port mapping in ``.env`` / compose.

Usage::

    python scripts/demo_api_requests.py
    python scripts/demo_api_requests.py --base-url http://127.0.0.1:8000 --only-tags suspicious
    python scripts/demo_api_requests.py --async --id-suffix _live

Expected console output: one line per request with HTTP status and a short JSON snippet
(e.g. ``200 … "prediction_label":"legit"`` or ``409 … already exists``).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _api_body(rec: dict, *, id_suffix: str) -> dict[str, Any]:
    tid = rec["transaction_id"] + id_suffix
    body: dict[str, Any] = {
        "transaction_id": tid,
        "customer_id": rec["customer_id"],
        "amount": rec["amount"],
        "merchant": rec.get("merchant"),
        "category": rec.get("category"),
        "country": rec.get("country"),
        "city": rec.get("city"),
        "device_type": rec.get("device_type"),
        "ip_address": rec.get("ip_address"),
        "transacted_at": rec["transacted_at"],
    }
    if rec.get("model_features") is not None:
        body["model_features"] = rec["model_features"]
    return body


def _post_json(url: str, payload: dict, timeout: float) -> tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        return e.code, (e.read() or b"").decode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Send demo fraud score requests to the API.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="API origin (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=Path("data/demo/transactions.json"),
        help="Demo JSON (default: data/demo/transactions.json)",
    )
    parser.add_argument(
        "--only-tags",
        default="",
        help="Comma-separated tags; if set, only rows whose ``tags`` contains any of these",
    )
    parser.add_argument("--limit", type=int, default=0, help="Max requests (0 = no limit)")
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="POST to /api/v1/fraud/score-async (202) instead of sync /score",
    )
    parser.add_argument(
        "--id-suffix",
        default="",
        help="Append to each transaction_id (avoids 409 after DB seed with same ids)",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    if not args.file.is_file():
        raise SystemExit(f"File not found: {args.file.resolve()}")

    raw = json.loads(args.file.read_text(encoding="utf-8"))
    rows: list[dict] = raw["transactions"]
    tag_filter = {t.strip() for t in args.only_tags.split(",") if t.strip()}

    path = "/api/v1/fraud/score-async" if args.use_async else "/api/v1/fraud/score"
    base = args.base_url.rstrip("/")
    url = f"{base}{path}"

    sent = 0
    for rec in rows:
        if tag_filter:
            tags = set(rec.get("tags") or [])
            if tags.isdisjoint(tag_filter):
                continue
        if args.limit and sent >= args.limit:
            break
        body = _api_body(rec, id_suffix=args.id_suffix)
        status, text = _post_json(url, body, args.timeout)
        sent += 1
        snippet = text.replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:217] + "..."
        print(f"{status:3d}  {body['transaction_id']!s}  {snippet}")


if __name__ == "__main__":
    main()
