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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

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


_K = TypeVar("_K")
_V = TypeVar("_V")


@dataclass
class CacheLayer(Generic[_K, _V]):
    """One single-flight cache: a value store + per-key locks + label + cap.

    Bundling these means ``store``/``locks`` can't drift out of sync at a call
    site, the metric ``name`` lives next to the data it labels, and a per-layer
    ``max_size`` opts into LRU eviction (unbounded otherwise — bounded
    naturally by the benchmark/model registries).
    """

    name: str
    store: OrderedDict[_K, _V]
    locks: dict[_K, asyncio.Lock] = field(default_factory=dict)
    max_size: int | None = None


# Schema cache — only kept because aggregators (build_model_scores,
# build_benchmark_leaders) scan ``summary.rows`` for individual models. Every
# other endpoint serialises once and throws the schema away.
_summary_schemas: CacheLayer[str, BenchmarkSummarySchema] = CacheLayer(
    name="summary_schema", store=OrderedDict()
)

# Serialised-bytes caches — one per endpoint, keyed by name (plus languages tuple
# for the lang-scoped summary variant). Bounded by registry size except for the
# language variant, which has an LRU cap.
_summary_bytes: CacheLayer[str, Serialized] = CacheLayer(
    name="summary", store=OrderedDict()
)
_summary_lang_bytes: CacheLayer[tuple[str, tuple[str, ...]], Serialized] = CacheLayer(
    name="summary_lang", store=OrderedDict(), max_size=256
)
_task_score_bytes: CacheLayer[str, Serialized] = CacheLayer(
    name="task_scores", store=OrderedDict()
)
_model_score_bytes: CacheLayer[str, Serialized] = CacheLayer(
    name="model_scores", store=OrderedDict()
)
_per_language_bytes: CacheLayer[str, Serialized] = CacheLayer(
    name="per_language", store=OrderedDict()
)


def _lock_for(locks: dict[_K, asyncio.Lock], key: _K) -> asyncio.Lock:
    lock = locks.get(key)
    if lock is None:
        lock = locks.setdefault(key, asyncio.Lock())
    return lock


async def _cache_or_build(
    layer: CacheLayer[_K, _V],
    key: _K,
    build: Callable[[], Awaitable[_V]],
) -> _V:
    """Generic single-flight cache-or-build.

    Reads ``layer.store[key]`` on the warm path; on miss, acquires the per-key
    lock, re-checks, then awaits ``build()`` and stores the result.
    ``layer.name`` labels the hit/miss counter so ops can split warm/cold by
    endpoint family. When ``layer.max_size`` is set, the oldest entry (and its
    lock) is evicted at the cap; unbounded stores still pay the
    ``move_to_end`` cost on hit but it's negligible at registry-bounded sizes.
    """
    store = layer.store
    locks = layer.locks

    def _record_hit(cached: _V) -> _V:
        store.move_to_end(key)
        CACHE_OUTCOMES.labels(layer=layer.name, outcome="hit").inc()
        return cached

    cached = store.get(key)
    if cached is not None:
        return _record_hit(cached)
    async with _lock_for(locks, key):
        cached = store.get(key)
        if cached is not None:
            return _record_hit(cached)
        CACHE_OUTCOMES.labels(layer=layer.name, outcome="miss").inc()
        value = await build()
        store[key] = value
        if layer.max_size is not None and len(store) > layer.max_size:
            evicted, _ = store.popitem(last=False)
            locks.pop(evicted, None)
        return value


async def _cached_bytes(
    layer: CacheLayer[_K, Serialized],
    key: _K,
    schema_builder: Callable[[], Awaitable[BaseModel]],
) -> Serialized:
    """Single-flight cache for serialised bytes.

    Wraps :func:`_cache_or_build` with the serialise-on-thread step:
    ``gzip.compress`` on a multi-MB body would otherwise pin the event loop.
    """

    async def _build_and_serialize() -> Serialized:
        schema = await schema_builder()
        return await asyncio.to_thread(serialize_schema, schema)

    return await _cache_or_build(layer, key, _build_and_serialize)


async def get_summary(name: str) -> BenchmarkSummarySchema:
    """Cached unfiltered summary schema for ``name``.

    Lang-scoped summaries are not exposed as schemas — only
    :func:`get_summary_bytes` builds them, and it discards the schema after
    serialisation.
    """

    async def _build() -> BenchmarkSummarySchema:
        logger.info("Building summary for %s", name)
        return await build_benchmark_summary(name, get_cache())

    return await _cache_or_build(_summary_schemas, name, _build)


async def get_summary_bytes(name: str, languages: tuple[str, ...] = ()) -> Serialized:
    """JSON bytes + gzip + ETag for the summary endpoint."""
    if not languages:
        return await _cached_bytes(_summary_bytes, name, lambda: get_summary(name))

    async def _build() -> BaseModel:
        logger.info("Building summary for %s (langs=%s)", name, languages)
        return await build_benchmark_summary(name, get_cache(), languages=languages)

    return await _cached_bytes(_summary_lang_bytes, (name, languages), _build)


async def get_per_language_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the per-language endpoint."""
    return await _cached_bytes(
        _per_language_bytes, name, lambda: build_benchmark_per_language(name)
    )


async def get_task_scores_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the task-scores endpoint."""

    async def _build() -> BaseModel:
        return await asyncio.to_thread(build_task_scores, name)

    return await _cached_bytes(_task_score_bytes, name, _build)


async def get_model_scores_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the model-scores endpoint."""
    return await _cached_bytes(
        _model_score_bytes, name, lambda: build_model_scores(name)
    )
