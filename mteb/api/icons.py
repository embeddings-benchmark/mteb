"""Benchmark-icon proxy with long-lived in-process cache.

Why proxy at all: upstream icons live on github.com which (a) issues a
redirect to ``raw.githubusercontent.com`` that is never cached and (b) serves
the SVG itself with ``Cache-Control: max-age=300``. That means every page
navigation re-pulls the same handful of flag SVGs. Proxying lets us serve
them once, then hand the browser ``max-age=31536000, immutable`` so subsequent
visits hit the local HTTP cache without even revalidating.

We keep the bytes in process rather than on disk — the entire icon corpus is
a few KB per benchmark, total under a couple of MB even at thousands of
entries. ``lru_cache`` would suffice but we want explicit eviction control
and a synchronous "is this cached?" peek for tests.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request
from dataclasses import dataclass
from urllib.error import URLError

logger = logging.getLogger(__name__)

# Upstream timeout. SVGs are tiny; if the fetch takes longer than this
# something's wrong on github's end and we'd rather 404 fast than block
# request handlers.
_FETCH_TIMEOUT_S = 5.0

# Max size we'll cache per icon. Defensive: any URL the registry advertises
# could in theory be huge. 2 MB comfortably fits the largest legitimate icons
# we've seen (ViDoRe V3's PNG is ~790 KB) while still catching misconfigured
# URLs before they balloon process memory.
_MAX_BYTES = 2 * 1024 * 1024  # 2MB

_DEFAULT_CONTENT_TYPE = "image/svg+xml"


@dataclass(frozen=True, slots=True)
class CachedIcon:
    """An icon's raw bytes + media type. ETag is handled by ETagMiddleware."""

    body: bytes
    content_type: str


_cache: dict[str, CachedIcon] = {}


def _fetch_sync(url: str) -> CachedIcon | None:
    """Pull an icon from ``url``. Returns ``None`` on any fetch failure.

    Failures are intentionally swallowed: a missing icon shouldn't break the
    benchmark card UI. The caller renders a placeholder when ``None``.
    """
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
    """Return the cached icon for ``name``, fetching from ``url`` on miss.

    Cache key is the benchmark name (not the URL): if mteb's icon URL ever
    changes for a benchmark, a server restart picks up the new one. Per-name
    keying also dedupes when multiple benchmarks happen to share a URL.
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
