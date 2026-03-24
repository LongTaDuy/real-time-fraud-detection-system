"""FastAPI application factory and ASGI entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.error_handlers import register_safe_error_handlers
from app.api.routes import fraud, health, predictions, transactions
from app.config import get_settings
from app.db.session import engine
from app.ml.inference import FraudPredictor
from app.observability import configure_observability, get_logger
from app.observability.middleware import RequestContextMiddleware
from app.observability.startup import log_startup_connectivity
from app.redis_client import close_redis
from app.security.body_size import MaxBodySizeMiddleware
from app.security.rate_limit import SlidingWindowRateLimitMiddleware

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        predictor = FraudPredictor.load(settings)
    except Exception:
        log.exception("model_load_fatal", detail="FraudPredictor.load failed; fraud routes will return 503")
        predictor = None
    app.state.fraud_predictor = predictor

    if predictor is not None:
        log.info(
            "model_load_complete",
            mode="stub" if predictor.is_stub else "bundle",
            model_version=predictor.reportable_model_version,
            bundle_path=str(settings.model_bundle_path) if settings.model_bundle_path else None,
        )
    else:
        log.error("model_unavailable", detail="fraud_predictor not loaded")

    log_startup_connectivity(settings)

    try:
        yield
    finally:
        engine.dispose()
        close_redis()


def create_app() -> FastAPI:
    settings = get_settings()
    configure_observability(settings)
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    register_safe_error_handlers(app)

    if settings.max_request_body_bytes > 0:
        app.add_middleware(
            MaxBodySizeMiddleware,
            max_content_length_bytes=settings.max_request_body_bytes,
        )

    if settings.api_rate_limit_per_minute > 0 or settings.fraud_post_rate_limit_per_minute > 0:
        app.add_middleware(SlidingWindowRateLimitMiddleware, settings=settings)

    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    app.include_router(fraud.router)
    app.include_router(transactions.router)
    app.include_router(predictions.router)
    return app


app = create_app()
