"""Process-level settings for the leaderboard API.

Environment variables (all optional):
* ``CORS_ORIGINS`` — comma-separated extra origins.
* ``PRELOAD`` — ``1`` to pre-build every summary in the background.
* ``CACHE_REPO`` — HF dataset id (default ``mteb/results``).
  Empty string disables hub load and forces a local cold rebuild.
* ``OG_DIR`` — OG hero PNG directory (default ``/data/og``).
* ``PREWARM_MAX_WORKERS`` — thread-pool size for the schema-cache prewarm
  (default 16). Tune down on small-memory hosts; tune up on large ones.
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
"""Public read-only service with no auth/secrets — open by default so
client-side OG previewers and share-card validators work cross-origin.
"""


class Settings(BaseSettings):
    """All env-var-tunable knobs for the API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # NoDecode disables pydantic-settings' JSON parsing so the validator
    # below sees the raw comma-separated string instead of a JSONDecodeError.
    cors_origins: Annotated[Sequence[str], NoDecode] = Field(
        default_factory=lambda: _DEFAULT_CORS_ORIGINS
    )
    preload: bool = False
    # Empty string disables hub load (cold rebuild)
    cache_repo: str | None = "mteb/results"
    og_dir: str = "/data/og"
    # Thread-pool size for prewarm_schema_caches; pydantic-core construction
    # releases the GIL so this scales with cores.
    prewarm_max_workers: int = Field(default=16, ge=1)

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
        """Parse a comma-separated string (or list), dedup. Empty falls back to default."""
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
    """Return a process-singleton `Settings` parsed from environment.

    Memoised because hot paths (cache loaders, app factory, OG mount) call
    this on every request; env is stable for the process lifetime.
    """
    return Settings()
