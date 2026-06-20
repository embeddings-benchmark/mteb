"""Prometheus instrumentation.

The middleware labels metrics by *resource group* (``/v1/benchmarks``,
``/v1/tasks``, …) rather than full route template, to keep cardinality bounded
by the number of endpoint families.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.routing import Match

if TYPE_CHECKING:
    from starlette.types import ASGIApp, Message, Receive, Scope, Send

# Tuned for ETag-cached read API: warm <50 ms, cold builds into seconds.
_LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

# 304 (tiny) through cold full summaries (a few MB pre-gzip).
_RESPONSE_SIZE_BUCKETS = (
    256,
    1_024,
    4_096,
    16_384,
    65_536,
    262_144,
    1_048_576,
    4_194_304,
    16_777_216,
)

# Dedicated registry avoids double-registration in tests.
REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests processed, labelled by method, resource group, and status code.",
    labelnames=("method", "handler", "status"),
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds, measured from middleware entry to response start.",
    labelnames=("method", "handler"),
    buckets=_LATENCY_BUCKETS,
    registry=REGISTRY,
)

RESPONSE_SIZE = Histogram(
    "http_response_size_bytes",
    "HTTP response body size in bytes (post-gzip when the client got gzip).",
    labelnames=("method", "handler", "status"),
    buckets=_RESPONSE_SIZE_BUCKETS,
    registry=REGISTRY,
)

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "In-flight HTTP requests, labelled by method and resource group.",
    labelnames=("method", "handler"),
    registry=REGISTRY,
)

EXCEPTIONS_TOTAL = Counter(
    "http_request_exceptions_total",
    "Unhandled exceptions raised while processing a request.",
    labelnames=("method", "handler", "exception"),
    registry=REGISTRY,
)

# Per-entity counters; only incremented after the handler confirms the name exists.
BENCHMARK_SELECTIONS = Counter(
    "mteb_benchmark_selections_total",
    "Times a benchmark was requested, by name (summed across endpoints).",
    labelnames=("name",),
    registry=REGISTRY,
)

TASK_SELECTIONS = Counter(
    "mteb_task_selections_total",
    "Times a task was requested, by name (summed across endpoints).",
    labelnames=("name",),
    registry=REGISTRY,
)

MODEL_SELECTIONS = Counter(
    "mteb_model_selections_total",
    "Times a model was requested, by name (summed across endpoints).",
    labelnames=("name",),
    registry=REGISTRY,
)

# Hit/miss per cache layer.
CACHE_OUTCOMES = Counter(
    "mteb_cache_total",
    "Cache hits and misses for the serialised-bytes layer.",
    labelnames=("layer", "outcome"),
    registry=REGISTRY,
)


_UNMATCHED = "<unmatched>"
_ROUTE_GROUP_KEY = "_mteb_route_group"


def _group(template: str) -> str:
    """Collapse a route template to its literal skeleton by dropping ``{...}`` segments."""
    parts = [
        p for p in template.split("/") if not (p.startswith("{") and p.endswith("}"))
    ]
    skeleton = "/".join(parts)
    return skeleton or template


def _resolve_route_template(scope: Scope) -> str:
    route = scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)
    app = scope.get("app")
    router = getattr(app, "router", None) if app is not None else None
    if router is None:
        return _UNMATCHED
    for candidate in router.routes:
        match, _ = candidate.matches(scope)
        if match == Match.FULL and hasattr(candidate, "path"):
            return str(candidate.path)
    return _UNMATCHED


def _route_group(scope: Scope) -> str:
    """Resolve the route's resource group, memoised on ``scope``.

    Why: starlette fills ``scope["route"]`` only after the router runs, so we
    re-run ``matches`` as a pre-handler fallback.
    """
    cached: str | None = scope.get(_ROUTE_GROUP_KEY)
    if cached is not None and cached != _UNMATCHED:
        return cached
    resolved = _resolve_route_template(scope)
    group = _UNMATCHED if resolved == _UNMATCHED else _group(resolved)
    scope[_ROUTE_GROUP_KEY] = group
    return group


class PrometheusMiddleware:
    """Records per-request Prometheus metrics."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Time the inner app and emit method/handler/status metrics."""
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        handler = _route_group(scope)
        matched = handler != _UNMATCHED
        in_progress = (
            REQUESTS_IN_PROGRESS.labels(method=method, handler=handler)
            if matched
            else None
        )
        if in_progress is not None:
            in_progress.inc()

        status_code = "500"
        response_bytes = 0

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, response_bytes
            if message["type"] == "http.response.start":
                status_code = str(message["status"])
            elif message["type"] == "http.response.body":
                body = message.get("body")
                if body:
                    response_bytes += len(body)
            await send(message)

        start = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        except BaseException as exc:
            EXCEPTIONS_TOTAL.labels(
                method=method,
                handler=handler,
                exception=type(exc).__name__,
            ).inc()
            REQUEST_COUNT.labels(
                method=method,
                handler=handler,
                status="500",
            ).inc()
            raise
        finally:
            if matched:
                REQUEST_LATENCY.labels(
                    method=method,
                    handler=handler,
                ).observe(time.perf_counter() - start)
            if in_progress is not None:
                in_progress.dec()
        # Re-resolve post-handler: pre-call lookup misses on first-time templates.
        final_handler = _route_group(scope)
        REQUEST_COUNT.labels(
            method=method,
            handler=final_handler,
            status=status_code,
        ).inc()
        if final_handler != _UNMATCHED:
            RESPONSE_SIZE.labels(
                method=method,
                handler=final_handler,
                status=status_code,
            ).observe(response_bytes)


def render_metrics() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the Prometheus exposition format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
