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


def _lock_for(locks: dict[_K, asyncio.Lock], key: _K) -> asyncio.Lock:
    lock = locks.get(key)
    if lock is None:
        lock = locks.setdefault(key, asyncio.Lock())
    return lock


async def _cached_bytes(
    store: dict[_K, Serialized],
    locks: dict[_K, asyncio.Lock],
    key: _K,
    schema_builder: Callable[[], Awaitable[BaseModel]],
    *,
    layer: str = "bytes",
) -> Serialized:
    """Generic single-flight cache-or-build for serialised bytes.

    Builds the schema under the per-key lock so concurrent cold requests share
    one schema build; serialises on a worker thread because ``gzip.compress``
    on a multi-MB body would otherwise pin the event loop. ``layer`` labels
    the hit/miss counter so ops can split warm/cold by endpoint family.
    """
    cached = store.get(key)
    if cached is not None:
        CACHE_OUTCOMES.labels(layer=layer, outcome="hit").inc()
        return cached
    async with _lock_for(locks, key):
        cached = store.get(key)
        if cached is not None:
            CACHE_OUTCOMES.labels(layer=layer, outcome="hit").inc()
            return cached
        CACHE_OUTCOMES.labels(layer=layer, outcome="miss").inc()
        schema = await schema_builder()
        cached = await asyncio.to_thread(serialize_schema, schema)
        store[key] = cached
        return cached


async def get_summary(name: str) -> BenchmarkSummarySchema:
    """Cached unfiltered summary schema for ``name``.

    Lang-scoped summaries are not exposed as schemas — only
    :func:`get_summary_bytes` builds them, and it discards the schema after
    serialisation.
    """
    cached = _summary_schemas.get(name)
    if cached is not None:
        return cached
    async with _lock_for(_summary_locks, name):
        cached = _summary_schemas.get(name)
        if cached is not None:
            return cached
        logger.info("Building summary for %s", name)
        schema = await build_benchmark_summary(name, get_cache())
        _summary_schemas[name] = schema
        return schema


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

    key = (name, languages)
    cached = _summary_lang_bytes.get(key)
    if cached is not None:
        _summary_lang_bytes.move_to_end(key)
        CACHE_OUTCOMES.labels(layer="summary_lang", outcome="hit").inc()
        return cached
    async with _lock_for(_summary_lang_bytes_locks, key):
        cached = _summary_lang_bytes.get(key)
        if cached is not None:
            _summary_lang_bytes.move_to_end(key)
            CACHE_OUTCOMES.labels(layer="summary_lang", outcome="hit").inc()
            return cached
        CACHE_OUTCOMES.labels(layer="summary_lang", outcome="miss").inc()
        logger.info("Building summary for %s (langs=%s)", name, languages)
        schema = await build_benchmark_summary(name, get_cache(), languages=languages)
        cached = await asyncio.to_thread(serialize_schema, schema)
        _summary_lang_bytes[key] = cached
        if len(_summary_lang_bytes) > _SUMMARY_LANG_MAX:
            evicted, _ = _summary_lang_bytes.popitem(last=False)
            _summary_lang_bytes_locks.pop(evicted, None)
        return cached


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
        return await asyncio.to_thread(build_task_scores, name, get_cache())

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
