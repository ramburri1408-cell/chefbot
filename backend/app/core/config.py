"""
app/core/config.py

Single source of truth for all configuration. Pydantic-settings reads from
environment variables and .env files, validates types, and raises on startup
if required values are missing — better to fail fast on bad config than fail
mysteriously at request time.
"""

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Anthropic
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-6"

    # Database
    database_url: str
    redis_url: str = "redis://localhost:6379/0"

    # Vector store
    vector_store: Literal["chroma", "pinecone"] = "chroma"
    chroma_path: str = "./data/chroma"
    pinecone_api_key: str = ""
    pinecone_index: str = "chefbot-menu"

    # Security
    jwt_secret: str
    allowed_origins: list[str] = ["http://localhost:3000"]

    # Rate limiting
    rate_limit_per_minute: int = 30
    max_concurrent_sessions: int = 10_000

    # App
    debug: bool = False
    log_level: str = "INFO"

    @field_validator("database_url")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith(("postgresql", "sqlite")):
            raise ValueError("DATABASE_URL must be a PostgreSQL or SQLite URL")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings — parsed once at startup, reused everywhere.
    The lru_cache ensures we don't re-parse from env on every request.
    """
    return Settings()
