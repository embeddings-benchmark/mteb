"""Prometheus instrumentation.

The ASGI middleware records request count / in-flight / latency per matched
route template (handler label is the template, not the raw URL, to keep
cardinality bounded). :func:`render_metrics` produces the exposition body.
"""

from __future__ import annotations

import time
from collections.abc import (  # noqa: TC003 — used at runtime by middleware signature
    Awaitable,
    Callable,
)

from fastapi import (  # noqa: TC002 — runtime use in middleware signature
    Request,
    Response,
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

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


def _resolve_route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)
    app = request.scope.get("app")
    router = getattr(app, "router", None) if app is not None else None
    if router is None:
        return _UNMATCHED
    for candidate in router.routes:
        match, _ = candidate.matches(request.scope)
        if match == Match.FULL and hasattr(candidate, "path"):
            return str(candidate.path)
    return _UNMATCHED


def _route_template(request: Request) -> str:
    """Return the matched route's path template, or ``"<unmatched>"``.

    Starlette only fills ``scope["route"]`` after the router runs — which for
    BaseHTTPMiddleware happens inside ``call_next``. As a fallback we re-run
    ``matches`` against the app's router; the result is memoised on the scope
    so the post-handler re-resolution doesn't repeat the scan.
    """
    cached = request.scope.get(_ROUTE_TEMPLATE_KEY)
    if cached is not None and cached != _UNMATCHED:
        return cached
    resolved = _resolve_route_template(request)
    request.scope[_ROUTE_TEMPLATE_KEY] = resolved
    return resolved


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record per-request metrics around the downstream handler.

    Unmatched (404) requests stay out of the latency histogram + in-flight gauge
    so bot scans don't pollute p95/p99, but their count still increments. The
    ``/metrics`` route is excluded entirely to keep scrapes off the series.
    """

    async def dispatch(  # noqa: PLR6301 — must be a method on BaseHTTPMiddleware
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Time the downstream handler and emit method/handler/status metrics."""
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        handler = _route_template(request)
        matched = handler != _UNMATCHED
        in_progress = (
            REQUESTS_IN_PROGRESS.labels(method=method, handler=handler)
            if matched
            else None
        )
        if in_progress is not None:
            in_progress.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
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
        # Re-resolve post-handler — catches routes only stamped onto
        # scope["route"] after call_next.
        REQUEST_COUNT.labels(
            method=method,
            handler=_route_template(request),
            status=str(response.status_code),
        ).inc()
        return response


def render_metrics() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the Prometheus exposition format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
