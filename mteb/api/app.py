"""FastAPI application factory.

Run with ``uvicorn mteb.api.app:app --reload --port 8000``.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from collections.abc import (  # noqa: TC003 — used at runtime by lifespan signature
    AsyncIterator,
)
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from mteb.api.cache import preload_summaries_in_background, warmup_blocking
from mteb.api.metrics import PrometheusMiddleware
from mteb.api.otel import setup_telemetry, shutdown_telemetry
from mteb.api.routes import infra_router, router
from mteb.api.settings import cors_origins, og_dir

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """ASGI lifespan: prepare caches before HTTP listener accepts traffic."""
    await asyncio.to_thread(warmup_blocking)
    preload_summaries_in_background()
    yield
    shutdown_telemetry()


def create_app() -> FastAPI:
    """Build and return the FastAPI app instance for the leaderboard API."""
    app = FastAPI(title="MTEB Leaderboard API", lifespan=lifespan)
    # OTEL instrumentation must attach before middleware finalises so the
    # FastAPI instrumentor can inject its ASGI middleware.
    setup_telemetry(app)
    # GZip compresses 200 responses that aren't already encoded (cached routes
    # set Content-Encoding: gzip themselves and skip recompression). CORS adds
    # headers next; Prometheus is the outermost middleware so it sees the full
    # stack timing.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["GET"],
        allow_headers=["*"],
        expose_headers=["ETag", "Cache-Control"],
    )
    app.add_middleware(PrometheusMiddleware)
    app.include_router(router, prefix="/v1")
    app.include_router(infra_router)
    _mount_og_assets(app)
    return app


def _mount_og_assets(app: FastAPI) -> None:
    """Mount the pre-rendered Open Graph hero PNG directory at ``/og``.

    Skipped silently when the directory is missing so local dev doesn't error
    before the operator has run the generator.
    """
    cache_dir = og_dir()
    if pathlib.Path(cache_dir).is_dir():
        app.mount(
            "/og",
            _CachedStatic(directory=cache_dir, check_dir=False),
            name="og",
        )
    else:
        logger.warning(
            "OG cache directory %s does not exist; /og will 404 until generated.",
            cache_dir,
        )


class _CachedStatic(StaticFiles):
    """``StaticFiles`` with a one-day ``Cache-Control`` on every 200 response.

    Filenames are content-hashed against rendering inputs, but the URL never
    rolls over — the short max-age caps how long crawlers serve stale images.
    """

    async def get_response(self, path, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        return response


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app)
