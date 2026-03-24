# Tests

## What runs here

- **API** (`TestClient`): health, fraud score (sync + async), transactions list/detail, validation errors.
- **Cache**: Redis contract via `fakeredis` + `CacheService` round-trip and invalid JSON handling.
- **Services**: `FraudScoringService` (DB-backed `transaction_id` check, then cache/DB replay, model error mapping, end-to-end with SQLite), `TransactionPersistenceService`, `TransactionReadService`.
- **ML**: `FraudPredictor.load` — stub when no/missing/bad bundle; real `sklearn` pipeline from a temp bundle; missing `model_features` columns.

Tests use **SQLite** (in-memory, `StaticPool`) and **fakeredis** so they stay deterministic and do not require Postgres or a live Redis server.

## Prerequisites

- **Python 3.11+** (matches the Docker image; 3.12/3.13 are typically fine).
- Application dependencies from `requirements.txt`, plus dev tools:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

If `pip install -r requirements-dev.txt` fails (for example on an unsupported Python version), install at least:

```bash
pip install pytest fakeredis
```

## Run

From the **repository root** (so `pytest.ini` and `pythonpath = .` apply):

```bash
python -m pytest
```

Useful options:

```bash
python -m pytest tests/ -q              # quiet
python -m pytest tests/test_fraud_score_api.py -v
python -m pytest tests/ --tb=short -x   # stop on first failure
```

## Environment

- `DATABASE_URL` and `REDIS_URL` are set by `tests/conftest.py` before imports when missing.
- The app configures **structlog** on `create_app()`; tests will emit structured lines to stdout (harmless).
- The `app_client` fixture clears `MODEL_BUNDLE_PATH` so the API uses the **stub** predictor unless a test sets the bundle path itself (see `test_inference_model_loading.py`).

If your shell exports `MODEL_BUNDLE_PATH` to a real bundle, inference tests still manipulate env and clear `get_settings` cache; API tests force stub mode via the fixture.

## Observability (local runs)

- **Request logs**: `request_completed` with `method`, `path`, `status_code`, `duration_ms`, plus `request_id` in context (also returned as `X-Request-ID`).
- **Fraud**: `fraud_score_completed` (latency, cache hit/miss, scores) and `fraud_score_rejected` for duplicate / payload mismatch / model input errors.
- **Startup**: `model_load_complete`, `postgres_connectivity_ok` / `_failed`, `redis_connectivity_ok` / `_failed`.
- Tune verbosity with env `LOG_LEVEL` (default `INFO`).

## Robustness defaults in tests

- `API_RATE_LIMIT_PER_MINUTE=0` and `FRAUD_POST_RATE_LIMIT_PER_MINUTE=0` disable in-process rate limits during pytest (see `tests/conftest.py`). In production, set positive values; limits are per API process (use a shared store if you scale horizontally).

## Docker (optional)

If you prefer not to install ML stack locally, run tests in a Linux container with the repo mounted:

```bash
docker run --rm -v "$(pwd):/app" -w /app python:3.11-slim bash -c \
  "pip install -q -r requirements.txt pytest fakeredis httpx && python -m pytest tests/ -q"
```

(First run may take several minutes while dependencies install.)
