"""OpenTelemetry tracing for the leaderboard API.

Configures a single :class:`TracerProvider` with an OTLP HTTP span
exporter and instruments the FastAPI app for per-request server spans.
Metrics and logs are intentionally **not** wired here — Prometheus
already handles metrics at ``/metrics`` and stdlib ``logging`` is left
alone, so the OTEL surface stays small.

The whole setup is a no-op when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
unset, so local dev and the test suite don't try to push to a
collector that isn't there.

Standard OTEL env vars apply on top of the gate:

* ``OTEL_EXPORTER_OTLP_ENDPOINT`` — collector base URL (e.g.
  ``http://localhost:4318``). Setting this turns tracing on.
* ``OTEL_SERVICE_NAME`` — defaults to ``mteb-api`` when unset.
* ``OTEL_RESOURCE_ATTRIBUTES`` — extra ``key=value,key=value`` pairs.
* ``OTEL_EXPORTER_OTLP_HEADERS`` — auth headers for hosted collectors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from mteb.api.settings import otel_endpoint, otel_service_name

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Holds the configured provider so the lifespan shutdown hook can flush
# it. ``None`` until setup_telemetry() runs and the OTLP endpoint is set.
_tracer_provider: TracerProvider | None = None


def setup_telemetry(app: FastAPI) -> None:
    """Initialise the tracer provider + FastAPI instrumentation.

    Safe to call multiple times: a second call short-circuits via the
    module-level provider singleton. When tracing is disabled (no OTLP
    endpoint), this is a logged no-op.
    """
    global _tracer_provider  # noqa: PLW0603 — single-process singleton by design

    endpoint = otel_endpoint()
    if endpoint is None:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT unset; tracing disabled.")
        return
    if _tracer_provider is not None:
        # Already initialised — re-instrumenting FastAPI would raise.
        return

    # ``OTEL_RESOURCE_ATTRIBUTES`` is still picked up directly by
    # :func:`Resource.create`, so its key/value pairs merge in here on
    # top of our service name.
    resource = Resource.create({SERVICE_NAME: otel_service_name()})

    # BatchSpanProcessor buffers spans and ships them out of band so the
    # request path never blocks on the collector being slow.
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(_tracer_provider)

    # FastAPI auto-instrumentation: per-request server span, parent
    # context extraction from incoming W3C ``traceparent`` headers, and
    # standard ``http.*`` semantic attributes.
    FastAPIInstrumentor.instrument_app(app, tracer_provider=_tracer_provider)

    logger.info(
        "OpenTelemetry tracing initialised (endpoint=%s, service=%s).",
        endpoint,
        otel_service_name(),
    )


def shutdown_telemetry() -> None:
    """Flush + shut down the tracer provider that was set up.

    Called from the FastAPI lifespan teardown so the BatchSpanProcessor
    gets a chance to drain its queue before the process exits. Without
    this, the last batch of spans is dropped on shutdown.
    """
    global _tracer_provider  # noqa: PLW0603 — single-process singleton by design

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
