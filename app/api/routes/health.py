from fastapi import APIRouter, Depends
from redis import Redis
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db, get_redis_client

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@router.get("/health/ready")
def ready(
    db: Session = Depends(get_db),
    redis_client: Redis = Depends(get_redis_client),
) -> dict[str, str]:
    db.execute(text("SELECT 1"))
    redis_client.ping()
    return {"status": "ready", "database": "connected", "redis": "connected"}
