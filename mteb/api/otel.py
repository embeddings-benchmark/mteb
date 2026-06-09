"""OpenTelemetry tracing.

Configures one `TracerProvider` + OTLP HTTP exporter and instruments
FastAPI for per-request spans. No-op when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is
unset. Metrics live at ``/metrics`` (Prometheus); logs stay on stdlib.
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

if TYPE_CHECKING:
    from fastapi import FastAPI

    from mteb.api.settings import Settings

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None


def setup_telemetry(app: FastAPI, settings: Settings) -> None:
    """Initialise the tracer provider + FastAPI instrumentation. Idempotent."""
    global _tracer_provider  # noqa: PLW0603 — single-process singleton by design

    if settings.otel_endpoint is None:
        logger.info("OTEL_EXPORTER_OTLP_ENDPOINT unset; tracing disabled.")
        return
    if _tracer_provider is not None:
        return
    resource = Resource.create({SERVICE_NAME: settings.otel_service_name})
    _tracer_provider = TracerProvider(resource=resource)
    _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(_tracer_provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=_tracer_provider)

    logger.info(
        "OpenTelemetry tracing initialised (endpoint=%s, service=%s).",
        settings.otel_endpoint,
        settings.otel_service_name,
    )


def shutdown_telemetry() -> None:
    """Drain the BatchSpanProcessor queue before process exit."""
    global _tracer_provider  # noqa: PLW0603 — single-process singleton by design

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        _tracer_provider = None
