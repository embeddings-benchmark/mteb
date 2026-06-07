"""Benchmark-icon proxy with long-lived in-process cache.

Upstream icons live on github.com (redirects to raw, ``max-age=300``).
Proxying lets us hand the browser ``max-age=31536000, immutable``.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request
from dataclasses import dataclass
from urllib.error import URLError

logger = logging.getLogger(__name__)

# SVGs are tiny — if a fetch exceeds this, 404 fast instead of blocking handlers.
_FETCH_TIMEOUT_S = 5.0

# Defensive cap on cached body size to bound process memory.
_MAX_BYTES = 2 * 1024 * 1024  # 2MB

_DEFAULT_CONTENT_TYPE = "image/svg+xml"


@dataclass(frozen=True, slots=True)
class CachedIcon:
    """An icon's raw bytes + media type. ETag is handled by ETagMiddleware."""

    body: bytes
    content_type: str


_cache: dict[str, CachedIcon] = {}


def _fetch_sync(url: str) -> CachedIcon | None:
    """Pull an icon; ``None`` on any failure so the caller renders a placeholder."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "mteb-api/icon-proxy"})  # noqa: S310 — URL is from a trusted registry
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as resp:  # noqa: S310 — URL is from a trusted registry
            body = resp.read(_MAX_BYTES + 1)
            if len(body) > _MAX_BYTES:
                logger.warning("Icon at %s exceeds %d bytes; dropping", url, _MAX_BYTES)
                return None
            content_type = (
                resp.headers.get("Content-Type", _DEFAULT_CONTENT_TYPE)
                .split(";")[0]
                .strip()
            )
            return CachedIcon(body=body, content_type=content_type)
    except (URLError, TimeoutError, OSError) as exc:
        logger.info("Failed to fetch icon %s: %s", url, exc)
        return None


async def get_icon(name: str, url: str) -> CachedIcon | None:
    """Cached icon for ``name``, fetched from ``url`` on miss.

    Keyed by name so a URL change picks up at server restart and multiple
    benchmarks sharing a URL dedupe.
    """
    cached = _cache.get(name)
    if cached is not None:
        return cached

    fetched = await asyncio.to_thread(_fetch_sync, url)
    if fetched is None:
        return None
    _cache[name] = fetched
    return fetched


def has_cached(name: str) -> bool:
    """Used by tests to assert cache-hit behavior without forcing a fetch."""
    return name in _cache


def cache_clear() -> None:
    """Used by tests to start from an empty cache."""
    _cache.clear()
