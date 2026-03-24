"""Application settings loaded from environment and optional ``.env`` (single source of truth)."""

import re
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    app_name: str = "fraud-detection-api"
    database_url: str
    redis_url: str
    fraud_cache_ttl_seconds: int = 300
    default_model_version: str = "stub-v1"
    model_bundle_path: str | None = None
    fraud_score_queue_key: str = "fraud:score:jobs"
    log_level: str = "INFO"

    # Robustness / abuse
    trust_proxy_headers: bool = False
    max_request_body_bytes: int = 65_536
    api_rate_limit_per_minute: int = 300
    fraud_post_rate_limit_per_minute: int = 60
    db_connect_timeout_seconds: int = 10
    db_pool_timeout_seconds: int = 30
    redis_socket_connect_timeout_seconds: float = 5.0
    redis_socket_timeout_seconds: float = 5.0

    @field_validator("model_bundle_path", mode="before")
    @classmethod
    def empty_bundle_path_to_none(cls, v: object) -> object:
        if v == "":
            return None
        return v

    @field_validator("database_url")
    @classmethod
    def database_url_supported(cls, v: str) -> str:
        s = v.strip()
        if not (s.startswith("postgresql") or s.startswith("sqlite")):
            raise ValueError(
                "DATABASE_URL must be a postgresql* or sqlite* SQLAlchemy URL (see project docs).",
            )
        return s

    @field_validator("redis_url")
    @classmethod
    def redis_url_supported(cls, v: str) -> str:
        s = v.strip()
        if not (s.startswith("redis://") or s.startswith("rediss://")):
            raise ValueError("REDIS_URL must start with redis:// or rediss://")
        return s

    @field_validator("log_level")
    @classmethod
    def log_level_allowed(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        u = v.strip().upper()
        if u not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return u

    @field_validator("fraud_cache_ttl_seconds")
    @classmethod
    def fraud_cache_ttl_sane(cls, v: int) -> int:
        if v < 30 or v > 604_800:
            raise ValueError("fraud_cache_ttl_seconds must be between 30 and 604800 (7 days)")
        return v

    @field_validator("fraud_score_queue_key")
    @classmethod
    def queue_key_safe(cls, v: str) -> str:
        s = v.strip()
        if not s or len(s) > 128:
            raise ValueError("fraud_score_queue_key must be 1-128 characters")
        if not re.fullmatch(r"[\w:.\-]+", s):
            raise ValueError("fraud_score_queue_key allows letters, digits, underscore, colon, dot, hyphen")
        return s

    @field_validator("max_request_body_bytes")
    @classmethod
    def body_limit_sane(cls, v: int) -> int:
        if v < 0 or v > 10_485_760:
            raise ValueError("max_request_body_bytes must be between 0 and 10485760")
        return v

    @field_validator("api_rate_limit_per_minute", "fraud_post_rate_limit_per_minute")
    @classmethod
    def rate_limit_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("rate limits must be >= 0 (0 disables that bucket)")
        return v


@lru_cache
def get_settings() -> Settings:
    """Return cached settings. Call ``get_settings.cache_clear()`` in tests when env changes."""
    return Settings()
