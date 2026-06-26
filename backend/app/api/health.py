"""
app/api/health.py

/health — liveness probe (is the process alive?)
/ready  — readiness probe (can the pod accept traffic?)

Kubernetes uses these differently:
  liveness:  if /health fails, k8s restarts the pod
  readiness: if /ready fails, k8s stops sending traffic to this pod
             without restarting it (used during rolling deploys)
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.db.redis import get_redis

router = APIRouter(tags=["infrastructure"])


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str
    redis: str


@router.get("/health", response_model=HealthResponse)
async def liveness():
    return {"status": "ok"}


@router.get("/ready", response_model=ReadyResponse)
async def readiness():
    redis_status = "unknown"
    try:
        redis = await get_redis()
        await redis.ping()
        redis_status = "ok"
    except Exception:
        redis_status = "unavailable"

    overall = "ok" if redis_status == "ok" else "degraded"
    return {"status": overall, "redis": redis_status}
