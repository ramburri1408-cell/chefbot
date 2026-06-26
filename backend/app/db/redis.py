"""
app/db/redis.py

Async Redis client with connection pooling.
A single pool is shared across all requests — creating a new connection
per request is expensive at scale.
"""

from typing import Optional

import redis.asyncio as aioredis

from app.core.config import get_settings

_redis: Optional[aioredis.Redis] = None


async def init_redis() -> None:
    global _redis
    settings = get_settings()
    _redis = aioredis.from_url(
        settings.redis_url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=50,
    )


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


async def get_redis() -> aioredis.Redis:
    if _redis is None:
        await init_redis()
    return _redis
