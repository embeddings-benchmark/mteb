"""FastAPI application factory.

Run with ``uvicorn mteb.api.app:app --reload --port 8000``.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import pathlib
from collections.abc import (  # noqa: TC003 — used at runtime by middleware signature
    AsyncIterator,
    Awaitable,
    Callable,
)
from contextlib import asynccontextmanager

import uvicorn
from fastapi import (  # noqa: TC002 — Request needed at runtime by middleware
    FastAPI,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from mteb.api.cache import preload_summaries_in_background, warmup_blocking
from mteb.api.metrics import PrometheusMiddleware
from mteb.api.otel import setup_telemetry, shutdown_telemetry
from mteb.api.routes import infra_router, router
from mteb.api.settings import cors_origins, og_dir

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1024)
def _etag_of(body: bytes) -> str:
    return '"' + hashlib.sha1(body, usedforsecurity=False).hexdigest() + '"'


class ETagMiddleware(BaseHTTPMiddleware):
    """Compute an ETag from the response body and serve 304 on ``If-None-Match``.

    Sits *above* :class:`GZipMiddleware` so the hash is taken from the
    uncompressed bytes — identical payloads produce identical ETags regardless
    of whether the client accepts gzip. Skips 304 short-circuit on non-2xx
    responses and on bodies that didn't come back fully buffered.
    """

    async def dispatch(  # noqa: PLR6301 — must be a method on BaseHTTPMiddleware
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Buffer the response, compute its ETag, and short-circuit to 304 on match."""
        response = await call_next(request)
        if (
            request.method != "GET"
            or not (200 <= response.status_code < 300)
            or response.headers.get("content-encoding")
        ):
            return response
        # Starlette wraps every downstream response as a streaming response
        # inside ``BaseHTTPMiddleware``, so ``body_iterator`` is always present
        # at runtime even though the static type is the bare ``Response``.
        body = b"".join([chunk async for chunk in response.body_iterator])  # type: ignore[attr-defined]
        etag = _etag_of(body)
        headers = dict(response.headers)
        headers["etag"] = etag
        headers.setdefault("cache-control", "no-cache")
        if request.headers.get("if-none-match") == etag:
            return Response(
                status_code=304,
                headers={"etag": etag, "cache-control": headers["cache-control"]},
            )
        headers["content-length"] = str(len(body))
        return Response(
            content=body,
            status_code=response.status_code,
            headers=headers,
            media_type=response.media_type,
        )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """ASGI lifespan: prepare caches before the HTTP listener accepts traffic.

    Replaces the deprecated ``@app.on_event("startup")`` hook with FastAPI's
    lifespan context manager. ``warmup_blocking`` is dispatched via
    :func:`asyncio.to_thread` so it can run its synchronous polars +
    pydantic-schema work without freezing the event loop — uvicorn waits
    for the ``yield`` before flipping the app into "ready" state, so by
    the time the first request arrives every shared cache is populated.
    """
    await asyncio.to_thread(warmup_blocking)
    preload_summaries_in_background()
    yield
    # Flush OTEL batch processors so the final spans / logs / metrics
    # reach the collector. Registry / pydantic caches die with the process.
    shutdown_telemetry()


def create_app() -> FastAPI:
    """Build and return the FastAPI app instance for the leaderboard API."""
    app = FastAPI(title="MTEB Leaderboard API", lifespan=lifespan)
    # OpenTelemetry instrumentation must be attached before the
    # middleware stack is finalised so the FastAPI instrumentor can
    # inject its ASGI middleware. No-op unless
    # OTEL_EXPORTER_OTLP_ENDPOINT is set.
    setup_telemetry(app)
    # ETag middleware first (outermost): runs after Gzip/CORS, so it hashes the
    # uncompressed body and the 304 short-circuit avoids paying any
    # downstream serialisation cost on revalidation. Gzip second so all
    # 200 responses get compressed. CORS last so its headers ride along.
    app.add_middleware(ETagMiddleware)
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["GET"],
        allow_headers=["*"],
        # Surface our caching headers to client-side JS so devtools / fetch
        # can confirm 304 revalidation is happening.
        expose_headers=["ETag", "Cache-Control"],
    )
    # Prometheus is added last so it ends up as the outermost middleware —
    # request latency is measured around the full middleware stack, and
    # every response (including CORS-modified or 304-short-circuited ones)
    # is counted. The /metrics scrape itself is excluded inside the
    # middleware to keep self-traffic out of the series.
    app.add_middleware(PrometheusMiddleware)
    # Versioned API surface (/v1/benchmarks, /v1/tasks, /v1/models, /v1/icon, …)
    # plus the infra endpoints at root (/health, /metrics, …).
    app.include_router(router, prefix="/v1")
    app.include_router(infra_router)
    _mount_og_assets(app)
    return app


def _mount_og_assets(app: FastAPI) -> None:
    """Mount the directory of pre-rendered Open Graph hero PNG files.

    The directory (``MTEB_API_OG_DIR``, default ``/data/og``) is populated
    out-of-band by ``scripts/generate_og_images.py`` — typically inside
    the Dockerfile's builder stage. Mounted with a long ``Cache-Control``
    because each filename is content-hashed against its rendering
    inputs: a real change writes a fresh hash sidecar, never a stale
    overwrite of an in-flight URL.

    Mount is skipped silently when the directory is missing — keeps
    local-dev startup green even before the operator has run the
    generator.
    """
    cache_dir = og_dir()
    if pathlib.Path(cache_dir).is_dir():
        # ``check_dir=False`` so the mount survives the directory being
        # empty at boot — the generator may populate it in-place without
        # an app restart.
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
    """``StaticFiles`` that adds a long ``Cache-Control`` to every response.

    The PNG filename embeds ``encodeURIComponent(entity_name)``; the
    generator overwrites in place when inputs change but the URL never
    rolls over. To keep CDNs and crawlers from serving forever-stale
    images on entity-data change, the sidecar hash flow handles
    invalidation via Cloudflare-style ``Cache-Control: public,
    max-age=86400`` — one day. Short enough that an updated card
    propagates within a day; long enough that crawlers don't re-fetch
    on every share.
    """

    async def get_response(self, path, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        if response.status_code == 200:
            response.headers.setdefault("Cache-Control", "public, max-age=86400")
        return response


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app)
