# Demo data

Curated **fintech-style** synthetic transactions for portfolio screenshots, interviews, and local smoke tests. Five rows are tagged `suspicious` (wire, crypto, gambling, velocity, foreign ATM).

## Files

| File | Purpose |
|------|---------|
| `transactions.json` | Default dataset (regenerate with `scripts/generate_demo_data.py`) |

## Scripts (run from repo root)

### 1. Generate JSON

```bash
python scripts/generate_demo_data.py
```

**Expected output (console):** one line, e.g.

```text
Wrote 24 transactions (5 tagged suspicious) -> C:\...\data\demo\transactions.json
```

### 2. Load into PostgreSQL

Requires `DATABASE_URL` (e.g. via `.env`). Apply migrations first (`alembic upgrade head` or start the stack so migrations ran).

```bash
python scripts/seed_postgres.py
python scripts/seed_postgres.py --skip-existing
```

**Expected output (console):** one `insert <transaction_id>` line per new row, optional `skip …` lines with `--skip-existing`, then:

```text
Done. Inserted 24, skipped 0, errors 0.
```

Re-run without `--skip-existing` after a full seed → process exits with **Duplicate transaction_id** for the first conflict.

### 3. Call the fraud scoring API

Start the API (e.g. `docker compose up api` or `uvicorn`). Default base URL is `http://127.0.0.1:8000`.

```bash
# All rows (sync score)
python scripts/demo_api_requests.py

# Only suspicious scenarios (good for a tight demo)
python scripts/demo_api_requests.py --only-tags suspicious

# After DB seed: same business ids return 409 — append a suffix for fresh scores
python scripts/demo_api_requests.py --id-suffix _api_demo

# Async enqueue (202 Accepted)
python scripts/demo_api_requests.py --async --only-tags suspicious
```

**Expected output (console):** one line per POST, e.g.

```text
200  txn_demo_gen_0000  {"transaction_id":"txn_demo_gen_0000","prediction_label":"legit",...}
409  txn_demo_susp_wire_round  {"detail":{"message":"...","transaction_id":"txn_demo_susp_wire_round"}}
202  txn_demo_susp_crypto_live  {"status":"accepted","transaction_id":"txn_demo_susp_crypto_live",...}
```

- **200** — sync `/score` succeeded; body includes model prediction fields from `FraudScoreApiResponse`.
- **409** — transaction_id already stored (e.g. after seed or a prior score).
- **202** — `/score-async` accepted; run the worker to drain the queue.
- **422** — validation or model input error (e.g. missing `model_features` if your bundle requires them).

## Suggested demo order

1. `docker compose up -d` (or start Postgres + Redis + API + worker as you prefer).
2. `python scripts/generate_demo_data.py` (optional if you keep the committed JSON).
3. `python scripts/seed_postgres.py` — fills `GET /api/v1/transactions` and predictions for screenshots.
4. `python scripts/demo_api_requests.py --only-tags suspicious --id-suffix _live` — shows live scoring without 409 conflicts.
