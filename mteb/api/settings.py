"""Process-level settings for the leaderboard API.

Environment variables (all optional):
* ``CORS_ORIGINS`` — comma-separated extra origins.
* ``PRELOAD`` — ``1`` to pre-build every summary in the background.
* ``CACHE_REPO`` — HF dataset id (default ``mteb/results``). Empty disables hub load.
* ``OG_DIR`` — OG hero PNG directory (default ``/data/og``).
* ``PREWARM_MAX_WORKERS`` — thread-pool size for schema-cache prewarm (default 16).
* ``PRELOAD_CONCURRENCY`` — concurrent summary builds during preload (default 4).
* ``LOG_LEVEL`` — root logging level (default ``INFO``).
* ``DISK_CACHE`` — ``1`` (default) to persist per-benchmark frames; ``0`` to always rebuild.
* ``HTTP_MAX_AGE`` — ``Cache-Control: max-age`` (seconds) for JSON endpoints; default 4h.
* ``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``OTEL_SERVICE_NAME`` — standard OTEL.
"""

from __future__ import annotations

import functools
from collections.abc import (
    Sequence,  # noqa: TC003 — pydantic evaluates field annotations at runtime
)
from typing import Annotated, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DEFAULT_CORS_ORIGINS: tuple[str, ...] = ("*",)
"""Public read-only API with no auth — open by default for cross-origin share-card previewers."""


class Settings(BaseSettings):
    """All env-var-tunable knobs for the API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # NoDecode: skip pydantic-settings' JSON parsing so the validator below
    # sees the raw comma-separated string.
    cors_origins: Annotated[Sequence[str], NoDecode] = Field(
        default_factory=lambda: _DEFAULT_CORS_ORIGINS
    )
    preload: bool = False
    cache_repo: str | None = "mteb/results"
    og_dir: str = "/data/og"
    prewarm_max_workers: int = Field(default=16, ge=1)
    preload_concurrency: int = Field(default=4, ge=1)
    log_level: str = Field(default="INFO")
    disk_cache: bool = True
    # 4h default; ETag drives 304 after expiry. ``0`` in dev for fresh data on refresh.
    http_max_age: int = Field(default=4 * 60 * 60, ge=0)

    otel_endpoint: str | None = Field(
        default=None,
        validation_alias="OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    otel_service_name: str = Field(
        default="mteb-api",
        validation_alias="OTEL_SERVICE_NAME",
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: Any) -> list[str]:
        """Parse a comma-separated string or list; empty falls back to default."""
        if isinstance(v, str):
            parts = [o.strip() for o in v.split(",") if o.strip()]
        elif isinstance(v, list):
            parts = [str(o).strip() for o in v if str(o).strip()]
        else:
            parts = []
        if not parts:
            return list(_DEFAULT_CORS_ORIGINS)
        return parts


@functools.cache
def get_settings() -> Settings:
    """Process-singleton `Settings` parsed from environment."""
    return Settings()
