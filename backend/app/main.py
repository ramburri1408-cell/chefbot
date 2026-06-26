"""
app/main.py

FastAPI application entry point.

Lifespan context manager handles startup/shutdown:
- Startup: connect to Redis, warm up vector store, verify DB migrations
- Shutdown: flush any pending writes, close connections cleanly

Middleware stack (order matters — applied bottom-up to requests):
  1. CORS — must be outermost
  2. Trusted hosts — security
  3. Rate limiting — before auth so we can limit even unauthenticated requests
  4. Request ID injection — for distributed tracing
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.admin import router as admin_router
from app.core.config import get_settings
from app.db.redis import init_redis, close_redis
from app.services.rag import get_vector_store

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    settings = get_settings()
    log.info("Starting ChefBot API (model=%s)", settings.model_name)

    # Connect Redis
    await init_redis()
    log.info("Redis connected")

    # Warm up vector store (triggers lazy init + index check)
    store = get_vector_store()
    log.info("Vector store ready (%s)", settings.vector_store)

    yield

    # Shutdown
    await close_redis()
    log.info("Shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="ChefBot API",
        description="AI-powered menu recommendation assistant for The Cheesecake Factory",
        version="1.0.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # CORS — must be first in stack
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Trusted host (prevents host-header injection)
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],
    )

    # Request ID middleware — injects x-request-id for distributed tracing
    @app.middleware("http")
    async def add_request_id(request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start      = time.perf_counter()
        response   = await call_next(request)
        duration   = round((time.perf_counter() - start) * 1000, 1)
        response.headers["x-request-id"] = request_id
        response.headers["x-duration-ms"] = str(duration)
        log.info(
            "%s %s %d %sms [%s]",
            request.method, request.url.path,
            response.status_code, duration, request_id,
        )
        return response

    # Routes
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(admin_router)

    return app


app = create_app()
