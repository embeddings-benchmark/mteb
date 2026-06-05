"""FastAPI application factory.

Run with ``uvicorn mteb.api.app:app --reload --port 8000``.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
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
from starlette.middleware.base import BaseHTTPMiddleware

from mteb.api.cache import preload_summaries_in_background, warmup_blocking
from mteb.api.metrics import PrometheusMiddleware
from mteb.api.otel import setup_telemetry, shutdown_telemetry
from mteb.api.routes import infra_router, router
from mteb.api.settings import cors_origins

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
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app)
