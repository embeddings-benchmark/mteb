"""Process-level settings for the leaderboard API.

Backed by :class:`pydantic_settings.BaseSettings` so env-var parsing,
type-coercion, and defaults live in one declarative block. Anything driven
by an environment variable lives here so the rest of the API package can
stay focused on logic. ``get_settings()`` is process-cached — call it any
number of times, the parse happens once.

Environment variables (all optional):

* ``MTEB_API_CORS_ORIGINS`` — comma-separated list of extra allowed origins.
  The four ``localhost``/``127.0.0.1`` dev ports are always allowed.
* ``MTEB_API_PRELOAD`` — set to ``1`` to pre-build every benchmark summary
  in the background warmup. Off by default (light warmup only).
* ``MTEB_API_CACHE_REPO`` — HF dataset id the leaderboard parquet cache
  is pulled from. Defaults to ``mteb/results``. Set to ``""`` (empty
  string) to disable the hub load and force a local cold rebuild.
* ``OTEL_EXPORTER_OTLP_ENDPOINT`` — collector base URL (e.g.
  ``http://localhost:4318``). When unset, OpenTelemetry is a no-op.
* ``OTEL_SERVICE_NAME`` — service name tagged on every signal. Defaults
  to ``mteb-api``.

The OTEL fields keep their standard ``OTEL_*`` names (no ``MTEB_API_``
prefix) so the same env vars also feed the OpenTelemetry SDK directly
for everything else it auto-discovers (resource attrs, headers, etc.).
"""

from __future__ import annotations

from collections.abc import (
    Sequence,  # noqa: TC003 — pydantic evaluates field annotations at runtime
)
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

_DEFAULT_CORS_ORIGINS: tuple[str, ...] = ("*",)
"""Open the API to every origin by default.

This is a public, read-only service — there's no auth, no secrets, and
the OG hero images are explicitly designed to be embedded cross-origin
from any social-media crawler or share-card validator. The narrower
"only the leaderboard frontend" allowlist that lived here previously
broke client-side OG previewers like opengraph.xyz with no
corresponding security gain (the data they were "guarding" is already
public). Operators who want a tighter list can set
``MTEB_API_CORS_ORIGINS`` — that value *replaces* this default, it
doesn't merge.
"""


class Settings(BaseSettings):
    """All env-var-tunable knobs for the API."""

    model_config = SettingsConfigDict(
        env_prefix="MTEB_API_",
        env_file=None,
        extra="ignore",
        populate_by_name=True,
    )

    # NoDecode disables pydantic-settings' default JSON parsing for the env
    # value so the field_validator below sees the raw comma-separated string
    # documented in the README / Dockerfile instead of a JSONDecodeError.
    cors_origins: Annotated[Sequence[str], NoDecode] = Field(
        default_factory=lambda: _DEFAULT_CORS_ORIGINS
    )
    preload: bool = False
    # ``MTEB_API_CACHE_REPO``. Empty string disables hub load entirely
    # and forces the cold-rebuild path. ``None`` means "use the default
    # repo id" (matches the env-unset case).
    cache_repo: str | None = "mteb/results"
    # ``MTEB_API_OG_DIR`` — filesystem path where the generated Open Graph
    # hero PNG files live. The startup mount uses this as the StaticFiles root
    # for ``/og``. Defaults to ``/data/og`` so deployments mount a
    # persistent volume at that path; in local dev override with
    # ``MTEB_API_OG_DIR=./.og-cache``.
    og_dir: str = "/data/og"

    # OTEL fields opt out of the ``MTEB_API_`` prefix via ``validation_alias``
    # so the standard ``OTEL_*`` env vars apply unchanged. That way the
    # OpenTelemetry SDK and our Settings read the *same* environment
    # variables, instead of forcing the operator to set both.
    #
    # An empty / unset endpoint means "telemetry disabled" — keeps local
    # dev and CI from spamming connection-refused errors at a non-existent
    # collector. The service name defaults to ``mteb-api`` so traces /
    # logs / metrics land in a sensible bucket even without explicit config.
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
        """Accept a comma-separated string (from env) or a list, dedup, return.

        Falls back to :data:`_DEFAULT_CORS_ORIGINS` (``["*"]``) when the
        env var is unset or parses to an empty list. Setting the env var
        *replaces* the default — operators who want to lock down to a
        specific allow-list spell it out in full.
        """
        if isinstance(v, str):
            parts = [o.strip() for o in v.split(",") if o.strip()]
        elif isinstance(v, list):
            parts = [str(o).strip() for o in v if str(o).strip()]
        else:
            parts = []
        if not parts:
            return list(_DEFAULT_CORS_ORIGINS)
        seen: set[str] = set()
        out: list[str] = []
        for o in parts:
            if o not in seen:
                seen.add(o)
                out.append(o)
        return out


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
    """Filesystem path that holds the generated Open Graph hero PNG files.

    The :func:`mteb.api.app.create_app` mount uses this as the
    ``StaticFiles`` directory for ``/og``. Caller-side code that
    needs to write into the directory (e.g. the ``generate.mjs``
    script invoked from CI) reads it from the env directly.
    """
    return get_settings().og_dir


def cache_repo() -> str:
    """HF dataset id that holds the leaderboard parquet cache.

    Empty string disables hub load (caller falls back to a local cold
    rebuild). Always returns a string — never ``None``.
    """
    val = get_settings().cache_repo
    return val if val is not None else ""


def otel_endpoint() -> str | None:
    """OTLP collector base URL, or ``None`` when telemetry should stay off."""
    val = get_settings().otel_endpoint
    return val.strip() if val and val.strip() else None


def otel_service_name() -> str:
    """``service.name`` resource attribute applied to every OTEL signal."""
    return get_settings().otel_service_name
