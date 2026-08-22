"""Benchmark-icon proxy with long-lived in-process cache.

Upstream icons have ``max-age=300``; proxying lets us serve ``immutable``.
"""

from __future__ import annotations

import asyncio
import logging
import time
import urllib.request
from dataclasses import dataclass
from urllib.error import URLError

logger = logging.getLogger(__name__)

# SVGs are tiny — fail fast instead of blocking handlers.
_FETCH_TIMEOUT_S = 5.0
_MAX_BYTES = 2 * 1024 * 1024
_DEFAULT_CONTENT_TYPE = "image/svg+xml"
# Negative cache TTL — keeps us off a flapping upstream.
_FAILURE_TTL_S = 60.0


@dataclass(frozen=True, slots=True)
class CachedIcon:
    """An icon's raw bytes + media type"""

    body: bytes
    content_type: str


_cache: dict[str, CachedIcon] = {}
_failure_cache: dict[str, float] = {}
_fetch_locks: dict[str, asyncio.Lock] = {}


def _lock_for(name: str) -> asyncio.Lock:
    return _fetch_locks.setdefault(name, asyncio.Lock())


def _fetch_sync(url: str) -> CachedIcon | None:
    """Pull an icon; ``None`` on any failure so the caller renders a placeholder."""
    try:
        req = urllib.request.Request(url)  # noqa: S310 — URL is from a trusted registry
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
    """Cached icon for ``name``; per-name lock single-flights concurrent cold requests."""
    cached = _cache.get(name)
    if cached is not None:
        return cached

    now = time.monotonic()
    failure_until = _failure_cache.get(name)
    if failure_until is not None and failure_until > now:
        return None

    async with _lock_for(name):
        cached = _cache.get(name)
        if cached is not None:
            return cached
        failure_until = _failure_cache.get(name)
        if failure_until is not None and failure_until > time.monotonic():
            return None

        fetched = await asyncio.to_thread(_fetch_sync, url)
        if fetched is None:
            _failure_cache[name] = time.monotonic() + _FAILURE_TTL_S
            return None
        _cache[name] = fetched
        _failure_cache.pop(name, None)
        return fetched
