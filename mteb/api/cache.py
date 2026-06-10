"""Per-endpoint pre-serialised bytes cache + single-flight cold build.

Holds the warm bytes routes serve directly. Cold builds run the schema
constructor under a per-key lock, then serialise + gzip on a worker thread.

Sits above :mod:`mteb.api.frames` and :mod:`mteb.api.serialization` and below
:mod:`mteb.api.routes` and :mod:`mteb.api.warmup`. Importing this module must
not pull in ``routes`` (warmup orchestration lives in :mod:`mteb.api.warmup`).
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING, TypeVar

from mteb.api.aggregators import (
    build_benchmark_per_language,
    build_benchmark_summary,
    build_model_scores,
    build_task_scores,
)
from mteb.api.frames import get_cache
from mteb.api.metrics import CACHE_OUTCOMES
from mteb.api.serialization import (  # noqa: TC001 — Serialized used as runtime dict value type
    Serialized,
    serialize_schema,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pydantic import BaseModel

    from mteb.api.schemas import BenchmarkSummarySchema

logger = logging.getLogger(__name__)


# Only kept because aggregators (build_model_scores, build_benchmark_leaders)
# scan ``summary.rows`` for individual models. Every other endpoint serialises
# once and throws the schema away.
_summary_schemas: dict[str, BenchmarkSummarySchema] = {}
_summary_locks: dict[str, asyncio.Lock] = {}

# Serialised-bytes caches — one per endpoint, keyed by name (plus languages tuple
# for the lang-scoped summary variant). Bounded by registry size except for the
# language variant, which has an LRU cap.
_SUMMARY_LANG_MAX = 256
_summary_bytes: dict[str, Serialized] = {}
_summary_lang_bytes: OrderedDict[tuple[str, tuple[str, ...]], Serialized] = (
    OrderedDict()
)
_task_score_bytes: dict[str, Serialized] = {}
_model_score_bytes: dict[str, Serialized] = {}
_per_language_bytes: dict[str, Serialized] = {}

_summary_bytes_locks: dict[str, asyncio.Lock] = {}
_summary_lang_bytes_locks: dict[tuple[str, tuple[str, ...]], asyncio.Lock] = {}
_task_score_bytes_locks: dict[str, asyncio.Lock] = {}
_model_score_bytes_locks: dict[str, asyncio.Lock] = {}
_per_language_bytes_locks: dict[str, asyncio.Lock] = {}

_K = TypeVar("_K")
_V = TypeVar("_V")


def _lock_for(locks: dict[_K, asyncio.Lock], key: _K) -> asyncio.Lock:
    lock = locks.get(key)
    if lock is None:
        lock = locks.setdefault(key, asyncio.Lock())
    return lock


async def _cache_or_build(
    store: dict[_K, _V] | OrderedDict[_K, _V],
    locks: dict[_K, asyncio.Lock],
    key: _K,
    build: Callable[[], Awaitable[_V]],
    *,
    layer: str,
    max_size: int | None = None,
) -> _V:
    """Generic single-flight cache-or-build.

    Reads ``store[key]`` on the warm path; on miss, acquires the per-key lock,
    re-checks, then awaits ``build()`` and stores the result. ``layer`` labels
    the hit/miss counter so ops can split warm/cold by endpoint family. When
    ``max_size`` is set, ``store`` must be an ``OrderedDict`` — hits bump LRU
    order and inserts evict the oldest entry (and its lock) at the cap.
    """
    cached = store.get(key)
    if cached is not None:
        if max_size is not None:
            store.move_to_end(key)  # type: ignore[union-attr]
        CACHE_OUTCOMES.labels(layer=layer, outcome="hit").inc()
        return cached
    async with _lock_for(locks, key):
        cached = store.get(key)
        if cached is not None:
            if max_size is not None:
                store.move_to_end(key)  # type: ignore[union-attr]
            CACHE_OUTCOMES.labels(layer=layer, outcome="hit").inc()
            return cached
        CACHE_OUTCOMES.labels(layer=layer, outcome="miss").inc()
        value = await build()
        store[key] = value
        if max_size is not None and len(store) > max_size:
            evicted, _ = store.popitem(last=False)  # type: ignore[call-arg]
            locks.pop(evicted, None)
        return value


async def _cached_bytes(
    store: dict[_K, Serialized] | OrderedDict[_K, Serialized],
    locks: dict[_K, asyncio.Lock],
    key: _K,
    schema_builder: Callable[[], Awaitable[BaseModel]],
    *,
    layer: str = "bytes",
    max_size: int | None = None,
) -> Serialized:
    """Single-flight cache for serialised bytes.

    Wraps :func:`_cache_or_build` with the serialise-on-thread step:
    ``gzip.compress`` on a multi-MB body would otherwise pin the event loop.
    """

    async def _build_and_serialize() -> Serialized:
        schema = await schema_builder()
        return await asyncio.to_thread(serialize_schema, schema)

    return await _cache_or_build(
        store, locks, key, _build_and_serialize, layer=layer, max_size=max_size
    )


async def get_summary(name: str) -> BenchmarkSummarySchema:
    """Cached unfiltered summary schema for ``name``.

    Lang-scoped summaries are not exposed as schemas — only
    :func:`get_summary_bytes` builds them, and it discards the schema after
    serialisation.
    """

    async def _build() -> BenchmarkSummarySchema:
        logger.info("Building summary for %s", name)
        return await build_benchmark_summary(name, get_cache())

    return await _cache_or_build(
        _summary_schemas, _summary_locks, name, _build, layer="summary_schema"
    )


async def get_summary_bytes(name: str, languages: tuple[str, ...] = ()) -> Serialized:
    """JSON bytes + gzip + ETag for the summary endpoint."""
    if not languages:
        return await _cached_bytes(
            _summary_bytes,
            _summary_bytes_locks,
            name,
            lambda: get_summary(name),
            layer="summary",
        )

    async def _build() -> BaseModel:
        logger.info("Building summary for %s (langs=%s)", name, languages)
        return await build_benchmark_summary(name, get_cache(), languages=languages)

    return await _cached_bytes(
        _summary_lang_bytes,
        _summary_lang_bytes_locks,
        (name, languages),
        _build,
        layer="summary_lang",
        max_size=_SUMMARY_LANG_MAX,
    )


async def get_per_language_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the per-language endpoint."""
    return await _cached_bytes(
        _per_language_bytes,
        _per_language_bytes_locks,
        name,
        lambda: build_benchmark_per_language(name),
        layer="per_language",
    )


async def get_task_scores_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the task-scores endpoint."""

    async def _build() -> BaseModel:
        return await asyncio.to_thread(build_task_scores, name)

    return await _cached_bytes(
        _task_score_bytes,
        _task_score_bytes_locks,
        name,
        _build,
        layer="task_scores",
    )


async def get_model_scores_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the model-scores endpoint."""
    return await _cached_bytes(
        _model_score_bytes,
        _model_score_bytes_locks,
        name,
        lambda: build_model_scores(name),
        layer="model_scores",
    )
