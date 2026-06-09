"""Process-wide caches for the leaderboard API.

Holds pre-serialised JSON bytes + gzipped bytes + ETag per endpoint. Routes
read straight from the bytes cache; cold builds run the schema constructor,
serialise on a worker thread (to keep gzip off the event loop), and store the
result.

Only ``_summary_schemas`` keeps the pydantic schema around — ``build_model_scores``
and ``build_benchmark_leaders`` scan its rows. Other endpoints discard the schema
after serialising.
"""

from __future__ import annotations

import asyncio
import functools
import gzip
import hashlib
import logging
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, cast

import polars as pl

from mteb.api.aggregators import build_benchmark_summary
from mteb.api.settings import get_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from concurrent.futures import Future

    from pydantic import BaseModel

    from mteb.api.schemas import BenchmarkSummarySchema
    from mteb.cache.result_cache import ResultCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Serialized:
    """Pre-serialised schema response — raw bytes + optional gzipped variant + ETag."""

    body: bytes
    body_gzip: bytes | None  # None when payload is too small to bother compressing
    etag: str


# Tiny payloads don't gain from gzip; framing alone is ~20 bytes of overhead.
_GZIP_MIN_BYTES = 1024


def serialize_bytes(body: bytes) -> Serialized:
    """Wrap raw JSON bytes with a gzipped variant (when worth it) + matching ETag.

    Synchronous — callers that hit this on a cold path should run it through
    :func:`asyncio.to_thread` so the gzip pass doesn't stall the event loop on
    multi-MB payloads.
    """
    body_gzip = (
        gzip.compress(body, compresslevel=6) if len(body) >= _GZIP_MIN_BYTES else None
    )
    etag = '"' + hashlib.sha1(body, usedforsecurity=False).hexdigest() + '"'
    return Serialized(body=body, body_gzip=body_gzip, etag=etag)


def _serialize(schema: BaseModel) -> Serialized:
    return serialize_bytes(schema.model_dump_json(by_alias=True).encode())


@functools.lru_cache(maxsize=1)
def get_cache() -> ResultCache:
    """Return the process-wide :class:`ResultCache`."""
    from mteb.cache.result_cache import ResultCache

    return ResultCache()


_UNIFIED_SCHEMA = {
    "model_name": pl.Utf8,
    "task_name": pl.Utf8,
    "subset": pl.Utf8,
    "score": pl.Float64,
}


def _dedupe_unified(combined: pl.DataFrame) -> pl.DataFrame:
    """Reduce the combined frame to one row per (model, task, subset) with max score."""
    if combined.is_empty():
        return pl.DataFrame(schema=_UNIFIED_SCHEMA)
    return (
        combined.drop_nulls("score")
        .group_by(["model_name", "task_name", "subset"])
        .agg(pl.col("score").max())
    )


_DEFAULT_CONFIG = "default"


def _load_default_from_hub(repo_id: str) -> pl.DataFrame | None:
    """Fetch the ``default`` HF config (all-results dump) via direct parquet read.

    Reads parquet shards with polars rather than ``datasets.load_dataset`` so a
    stale README ``dataset_info`` block (e.g. when a new column lands in the
    parquet but the YAML wasn't refreshed) doesn't fail the cast.
    """
    try:
        return pl.read_parquet(f"hf://datasets/{repo_id}/data/train-*.parquet")
    except Exception as exc:
        logger.warning("Hub load failed for %s: %s", repo_id, exc)
        try:
            from datasets import load_dataset

            return cast(
                "pl.DataFrame",
                load_dataset(repo_id, name=_DEFAULT_CONFIG, split="train").to_polars(),
            )
        except Exception as exc2:
            logger.warning(
                "Hub fallback also failed for %s/%s: %s", repo_id, _DEFAULT_CONFIG, exc2
            )
            return None


@functools.lru_cache(maxsize=1)
def _load_per_benchmark_frames() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return ``(per_benchmark_frames, unified_frame)``, loaded once."""
    import mteb
    from mteb.results.benchmark_results import BenchmarkResults

    cache = get_cache()
    combined: pl.DataFrame | None = None

    repo_id = get_settings().cache_repo
    if repo_id:
        combined = _load_default_from_hub(repo_id)
        if combined is not None and not combined.is_empty():
            logger.info(
                "Loaded default config (%d rows) from hub '%s'",
                combined.height,
                repo_id,
            )
        else:
            combined = None

    if combined is None:
        logger.info("Building combined results frame from local ResultCache")
        all_results = cache._load_from_cache(rebuild=True)
        combined = all_results._to_results_df()

    loaded = BenchmarkResults.split_leaderboard_frame(combined)
    for bench in mteb.get_benchmarks():
        loaded.setdefault(bench.name, pl.DataFrame(schema=combined.schema))

    unified = _dedupe_unified(combined)
    logger.info(
        "Built unified results frame: %d rows / %d unique tasks",
        unified.height,
        unified.select(pl.col("task_name").n_unique()).item() if unified.height else 0,
    )
    return loaded, unified


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
) -> Serialized:
    """Generic single-flight cache-or-build for serialised bytes.

    Builds the schema under the per-key lock so concurrent cold requests share
    one schema build; serialises on a worker thread because ``gzip.compress``
    on a multi-MB body would otherwise pin the event loop.
    """
    cached = store.get(key)
    if cached is not None:
        return cached
    async with _lock_for(locks, key):
        cached = store.get(key)
        if cached is not None:
            return cached
        schema = await schema_builder()
        cached = await asyncio.to_thread(_serialize, schema)
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
            _summary_bytes, _summary_bytes_locks, name, lambda: get_summary(name)
        )

    key = (name, languages)
    cached = _summary_lang_bytes.get(key)
    if cached is not None:
        _summary_lang_bytes.move_to_end(key)
        return cached
    async with _lock_for(_summary_lang_bytes_locks, key):
        cached = _summary_lang_bytes.get(key)
        if cached is not None:
            _summary_lang_bytes.move_to_end(key)
            return cached
        logger.info("Building summary for %s (langs=%s)", name, languages)
        schema = await build_benchmark_summary(name, get_cache(), languages=languages)
        cached = await asyncio.to_thread(_serialize, schema)
        _summary_lang_bytes[key] = cached
        if len(_summary_lang_bytes) > _SUMMARY_LANG_MAX:
            evicted, _ = _summary_lang_bytes.popitem(last=False)
            _summary_lang_bytes_locks.pop(evicted, None)
        return cached


async def get_per_language_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the per-language endpoint."""
    from mteb.api.aggregators import build_benchmark_per_language

    return await _cached_bytes(
        _per_language_bytes,
        _per_language_bytes_locks,
        name,
        lambda: build_benchmark_per_language(name),
    )


async def get_task_scores_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the task-scores endpoint."""
    from mteb.api.aggregators import build_task_scores

    async def _build() -> BaseModel:
        return await asyncio.to_thread(build_task_scores, name, get_cache())

    return await _cached_bytes(_task_score_bytes, _task_score_bytes_locks, name, _build)


async def get_model_scores_bytes(name: str) -> Serialized:
    """JSON bytes + gzip + ETag for the model-scores endpoint."""
    from mteb.api.aggregators import build_model_scores

    return await _cached_bytes(
        _model_score_bytes,
        _model_score_bytes_locks,
        name,
        lambda: build_model_scores(name),
    )


def _prewarm_training_datasets() -> None:
    """Populate ``_training_datasets_cached`` for every registered model.

    Without this, the first summary build pays ~2.5s on ``_collect_similar_tasks``
    per first-seen model.
    """
    from concurrent.futures import ThreadPoolExecutor

    from mteb.benchmarks._create_table import _training_datasets_cached
    from mteb.models.model_implementations import MODEL_REGISTRY

    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="warm-td") as ex:
        list(ex.map(_training_datasets_cached, MODEL_REGISTRY))


def _prewarm_list_schemas() -> None:
    """Pre-build the unfiltered list schemas + their serialised bytes.

    The four builders are independent — run them on a small thread pool.
    """
    from concurrent.futures import ThreadPoolExecutor

    from mteb.api.routes import (
        _benchmark_schemas_bytes,
        _filtered_model_schemas_bytes,
        _filtered_task_schemas_bytes,
        _menu_schemas_bytes,
    )

    with ThreadPoolExecutor(max_workers=4, thread_name_prefix="warm-list") as ex:
        futures = [
            ex.submit(_menu_schemas_bytes),
            ex.submit(_benchmark_schemas_bytes),
            ex.submit(_filtered_task_schemas_bytes, None, None, None, None, None),
            ex.submit(
                _filtered_model_schemas_bytes,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                False,
            ),
        ]
        for f in futures:
            f.result()


def warmup_blocking() -> None:
    """Populate shared caches synchronously before accepting requests.

    Phases 1-3 are independent (network I/O / CPU / pydantic-core); they run
    on a small thread pool so wall time is max() instead of sum(). Phase 4
    (list schemas) depends on phases 1-3 because it reads them through the
    benchmark / task / model caches, so it runs serially afterwards.
    """
    from mteb.api.adapters import prewarm_schema_caches

    try:
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="warmup") as ex:
            futures: list[Future[object]] = [
                ex.submit(_load_per_benchmark_frames),
                ex.submit(_prewarm_training_datasets),
                ex.submit(prewarm_schema_caches),
            ]
            for f in futures:
                f.result()
        _prewarm_list_schemas()
    except Exception as exc:
        logger.warning("light warmup failed: %s", exc)


def preload_summaries_in_background() -> None:
    """Pre-build every benchmark summary + per-language schema on a daemon thread."""
    if not get_settings().preload:
        return

    def _run() -> None:
        import mteb

        # Preload every registered benchmark — including off-menu
        # ones — so hidden-benchmark requests also hit the warmed cache
        # and the model detail page (which iterates all benchmarks) is
        # cold-path-free.
        all_names = [b.name for b in mteb.get_benchmarks()]

        async def _build_one(name: str) -> None:
            try:
                await get_summary_bytes(name)
            except Exception as exc:
                logger.warning("warmup summary: %s failed (%s)", name, exc)
            try:
                await get_per_language_bytes(name)
            except Exception as exc:
                logger.warning("warmup per-language: %s failed (%s)", name, exc)

        async def _build_all() -> None:
            await asyncio.gather(*(_build_one(n) for n in all_names))

        asyncio.run(_build_all())

    threading.Thread(target=_run, name="mteb-api-preload", daemon=True).start()
