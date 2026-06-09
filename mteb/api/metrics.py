"""Prometheus instrumentation.

The ASGI middleware records request count / in-flight / latency per matched
route template (handler label is the template, not the raw URL, to keep
cardinality bounded). :func:`render_metrics` produces the exposition body.
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

# Tuned for an ETag-cached read API: warm <50 ms, cold builds into seconds.
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

# Dedicated registry avoids double-registration in tests and leaks from other libs.
REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests processed, labelled by method, route template, and status code.",
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

REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "In-flight HTTP requests, labelled by method and route template.",
    labelnames=("method", "handler"),
    registry=REGISTRY,
)

EXCEPTIONS_TOTAL = Counter(
    "http_request_exceptions_total",
    "Unhandled exceptions raised while processing a request.",
    labelnames=("method", "handler", "exception"),
    registry=REGISTRY,
)

# Per-entity counters; cardinality bounded by the registries. Only incremented
# after the handler confirms the name exists.
BENCHMARK_SELECTIONS = Counter(
    "mteb_benchmark_selections_total",
    "Times a benchmark was requested, by name and endpoint kind.",
    labelnames=("name", "endpoint"),
    registry=REGISTRY,
)

TASK_SELECTIONS = Counter(
    "mteb_task_selections_total",
    "Times a task was requested, by name and endpoint kind.",
    labelnames=("name", "endpoint"),
    registry=REGISTRY,
)

MODEL_SELECTIONS = Counter(
    "mteb_model_selections_total",
    "Times a model was requested, by name and endpoint kind.",
    labelnames=("name", "endpoint"),
    registry=REGISTRY,
)


_UNMATCHED = "<unmatched>"
_ROUTE_TEMPLATE_KEY = "_mteb_route_template"


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


def _route_template(scope: Scope) -> str:
    """Return the matched route's path template, or ``"<unmatched>"``.

    Starlette only fills ``scope["route"]`` after the router runs. As a fallback
    we re-run ``matches`` against the app's router; the result is memoised on
    the scope so a post-handler re-resolution doesn't repeat the scan.
    """
    cached: str | None = scope.get(_ROUTE_TEMPLATE_KEY)
    if cached is not None and cached != _UNMATCHED:
        return cached
    resolved = _resolve_route_template(scope)
    scope[_ROUTE_TEMPLATE_KEY] = resolved
    return resolved


class PrometheusMiddleware:
    """Pure-ASGI middleware that records per-request Prometheus metrics.

    Unmatched (404) requests stay out of the latency histogram + in-flight gauge
    so bot scans don't pollute p95/p99, but their count still increments. The
    ``/metrics`` route and non-HTTP scopes (lifespan, websocket) are short-
    circuited so scrapes and ASGI control messages stay off the series.

    Implemented as raw ASGI (``__call__(scope, receive, send)``) rather than
    :class:`BaseHTTPMiddleware` because the latter buffers the response body
    through an internal queue — adds ~1 ms per request and breaks streaming.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Time the inner app and emit method/handler/status metrics.

        Short-circuits non-HTTP scopes (lifespan / websocket) and the
        ``/metrics`` scrape so neither pollutes the series. The ``send``
        callable is wrapped to capture the response status from the
        ``http.response.start`` ASGI message; status defaults to ``"500"`` so
        that a downstream crash before any response is sent still records.
        """
        if scope["type"] != "http" or scope["path"] == "/metrics":
            await self.app(scope, receive, send)
            return

        method = scope["method"]
        handler = _route_template(scope)
        matched = handler != _UNMATCHED
        in_progress = (
            REQUESTS_IN_PROGRESS.labels(method=method, handler=handler)
            if matched
            else None
        )
        if in_progress is not None:
            in_progress.inc()

        status_code = "500"

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = str(message["status"])
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
            REQUEST_COUNT.labels(method=method, handler=handler, status="500").inc()
            raise
        finally:
            if matched:
                REQUEST_LATENCY.labels(method=method, handler=handler).observe(
                    time.perf_counter() - start
                )
            if in_progress is not None:
                in_progress.dec()
        # Re-resolve post-handler — scope["route"] is only stamped after the
        # router runs, so the pre-call lookup misses on first-time templates.
        REQUEST_COUNT.labels(
            method=method,
            handler=_route_template(scope),
            status=status_code,
        ).inc()


def render_metrics() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the Prometheus exposition format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
