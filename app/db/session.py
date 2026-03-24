"""SQLAlchemy engine and request-scoped session factory for the API."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_settings = get_settings()

_connect_args: dict = {}
_engine_kwargs: dict = {"pool_pre_ping": True}
if _settings.database_url.startswith("postgresql"):
    _connect_args["connect_timeout"] = _settings.db_connect_timeout_seconds
    _engine_kwargs["pool_timeout"] = _settings.db_pool_timeout_seconds

engine = create_engine(
    _settings.database_url,
    connect_args=_connect_args,
    **_engine_kwargs,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Yield one session per HTTP request; close after the response is sent."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
