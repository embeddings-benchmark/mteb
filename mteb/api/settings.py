"""Process-level settings for the leaderboard API.

Environment variables (all optional):
* ``MTEB_API_CORS_ORIGINS`` — comma-separated extra origins.
* ``MTEB_API_PRELOAD`` — ``1`` to pre-build every summary in the background.
* ``MTEB_API_CACHE_REPO`` — HF dataset id (default ``mteb/results``).
  Empty string disables hub load and forces a local cold rebuild.
* ``MTEB_API_OG_DIR`` — OG hero PNG directory (default ``/data/og``).
* ``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``OTEL_SERVICE_NAME`` — standard OTEL.
"""

from __future__ import annotations

from collections.abc import (
    Sequence,  # noqa: TC003 — pydantic evaluates field annotations at runtime
)
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DEFAULT_CORS_ORIGINS: tuple[str, ...] = ("*",)
"""Public read-only service with no auth/secrets — open by default so
client-side OG previewers and share-card validators work cross-origin.
Operators who want a tighter list set ``MTEB_API_CORS_ORIGINS``, which
*replaces* this default (doesn't merge).
"""


class Settings(BaseSettings):
    """All env-var-tunable knobs for the API."""

    model_config = SettingsConfigDict(
        env_prefix="MTEB_API_",
        env_file=None,
        extra="ignore",
        populate_by_name=True,
    )

    # NoDecode disables pydantic-settings' JSON parsing so the validator
    # below sees the raw comma-separated string instead of a JSONDecodeError.
    cors_origins: Annotated[Sequence[str], NoDecode] = Field(
        default_factory=lambda: _DEFAULT_CORS_ORIGINS
    )
    preload: bool = False
    # Empty string disables hub load (cold rebuild). None ⇒ use default repo id.
    cache_repo: str | None = "mteb/results"
    og_dir: str = "/data/og"

    # OTEL_* env vars opt out of the MTEB_API_ prefix via validation_alias
    # so the OpenTelemetry SDK and our Settings read the same variables.
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
    def _parse_cors_origins(cls, v: object) -> list[str]:
        """Parse a comma-separated string (or list), dedup. Empty falls back to default."""
        from mteb.api.schemas import _dedupe_strs

        if isinstance(v, str):
            parts = [o.strip() for o in v.split(",") if o.strip()]
        elif isinstance(v, list):
            parts = [str(o).strip() for o in v if str(o).strip()]
        else:
            parts = []
        if not parts:
            return list(_DEFAULT_CORS_ORIGINS)
        return _dedupe_strs(parts)


def get_settings() -> Settings:
    """Return a freshly-parsed :class:`Settings` instance from the current environment."""
    return Settings()


def cors_origins() -> Sequence[str]:
    """Convenience accessor for the resolved CORS origin list."""
    return get_settings().cors_origins


def preload_full() -> bool:
    """Return ``True`` when ``MTEB_API_PRELOAD=1`` is set."""
    return get_settings().preload


def og_dir() -> str:
    """OG hero PNG directory (StaticFiles root for ``/og``)."""
    return get_settings().og_dir


def cache_repo() -> str:
    """HF dataset id for the leaderboard parquet cache; ``""`` disables hub load."""
    val = get_settings().cache_repo
    return val if val is not None else ""


def otel_endpoint() -> str | None:
    """OTLP collector URL, or ``None`` to keep telemetry off."""
    val = get_settings().otel_endpoint
    return val.strip() if val and val.strip() else None


def otel_service_name() -> str:
    """``service.name`` resource attribute applied to every OTEL signal."""
    return get_settings().otel_service_name
