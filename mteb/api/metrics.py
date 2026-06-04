"""Prometheus instrumentation for the leaderboard API.

Defines the metric collectors and an ASGI middleware that records request
count, in-flight requests, and request latency per matched route. Route
templates (e.g. ``/benchmarks/{name:path}/scores``) are used as the
``handler`` label so cardinality stays bounded — raw URL paths would
explode the time series count by every benchmark / task / model name.

Use :func:`render_metrics` to render the exposition body for the
``/metrics`` endpoint.
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

# Buckets tuned for an ETag-cached read API: most warm requests finish under
# 50 ms, cold benchmark builds can spike into the seconds. Wider than the
# Prometheus default so the long tail isn't squashed into ``+Inf``.
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

# A dedicated registry keeps the API's series out of the global default —
# avoids double-registration in tests and any future in-process Prometheus
# collectors from other libraries leaking in.
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

# Per-entity selection counters. Cardinality is bounded by the registries
# (one ``name`` label value per registered benchmark / task / model). Only
# incremented after the route handler has confirmed the name exists, so
# bot scans of unknown names don't balloon the series count.
#
# The ``endpoint`` label distinguishes the kind of view: ``detail`` for
# bare metadata, ``scores`` for the heavy scores payload, and ``icon`` for
# the benchmark icon proxy. That lets queries answer "top viewed scores
# pages" separately from "top metadata lookups".
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


def _route_template(request: Request) -> str:
    """Return the matched route's path template, or ``"<unmatched>"``.

    Starlette only fills ``scope["route"]`` after the router runs — which for
    BaseHTTPMiddleware happens inside ``call_next``. As a fallback we re-run
    ``matches`` against the app's router, which is what the router itself
    does and is cheap (a few regex tests).
    """
    route = request.scope.get("route")
    if route is not None and hasattr(route, "path"):
        return str(route.path)
    app = request.scope.get("app")
    router = getattr(app, "router", None) if app is not None else None
    if router is None:
        return "<unmatched>"
    for candidate in router.routes:
        match, _ = candidate.matches(request.scope)
        if match == Match.FULL and hasattr(candidate, "path"):
            return str(candidate.path)
    return "<unmatched>"


_UNMATCHED = "<unmatched>"


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record per-request Prometheus metrics around the downstream handler.

    Records four series:

    * ``http_requests_total{method, handler, status}`` — incremented once per response.
    * ``http_request_duration_seconds{method, handler}`` — observed once per response.
    * ``http_requests_in_progress{method, handler}`` — gauged for the call duration.
    * ``http_request_exceptions_total{method, handler, exception}`` — counted on raise,
      then re-raised so the standard ASGI 500 response still goes out.

    Unmatched (404) requests are kept out of the latency histogram and the
    in-flight gauge — bot scans of random URLs would otherwise pollute the
    p95/p99 latency story — but their *count* is still recorded under
    ``handler="<unmatched>"`` so scan volume stays visible. ``/metrics``
    itself is excluded entirely so Prometheus scrapes don't show up as
    application traffic.
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
        # Only matched routes contribute to in-flight / latency. Unmatched
        # paths still get a count below so 404 rate stays visible.
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
        # Re-resolve the template post-handler — for routes that only got
        # stamped onto ``scope["route"]`` after ``call_next``, this catches
        # them. ``<unmatched>`` paths still get counted; only the histogram
        # was skipped.
        REQUEST_COUNT.labels(
            method=method,
            handler=_route_template(request),
            status=str(response.status_code),
        ).inc()
        return response


def render_metrics() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the Prometheus exposition format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
