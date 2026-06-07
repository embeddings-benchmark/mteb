"""Process-wide caches for the leaderboard API.

Holds both pydantic schemas and their pre-serialised JSON bytes + ETag. Routes
that hit a warm cache return the bytes directly; cold builds run the schema
constructor once, then serialise once.
"""

from __future__ import annotations

import asyncio
import functools
import gzip
import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from mteb.api.aggregators import build_benchmark_summary
from mteb.api.settings import cache_repo, preload_full

if TYPE_CHECKING:
    from pydantic import BaseModel

    from mteb.api.schemas import (
        BenchmarkPerLanguageSchema,
        BenchmarkSummarySchema,
        ModelScoresSchema,
        TaskScoresSchema,
    )
    from mteb.cache.result_cache import ResultCache

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Serialized:
    """Pre-serialised schema response — raw + gzipped JSON bytes + ETag."""

    body: bytes
    body_gzip: bytes
    etag: str


# Skip the gzip variant for tiny payloads — gzip framing adds overhead and
# clients on a fast link don't gain anything from compressing a few KB.
_GZIP_MIN_BYTES = 1024


def serialize_bytes(body: bytes) -> Serialized:
    """Wrap raw JSON bytes with a gzipped variant + matching ETag."""
    body_gzip = (
        gzip.compress(body, compresslevel=6) if len(body) >= _GZIP_MIN_BYTES else body
    )
    etag = '"' + hashlib.sha1(body, usedforsecurity=False).hexdigest() + '"'
    return Serialized(body=body, body_gzip=body_gzip, etag=etag)


def _serialize(schema: BaseModel) -> Serialized:
    """Dump a schema to JSON, gzip-compress, and compute its ETag."""
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

            return load_dataset(
                repo_id, name=_DEFAULT_CONFIG, split="train"
            ).to_polars()
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

    repo_id = cache_repo()
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


# Schema caches. Bounded by the registry size for benchmarks/tasks/models;
# _summary_lang_schemas keys by (name, sorted_langs) so clients can in
# principle expand it — gated by an LRU eviction limit.
_summary_schemas: dict[str, BenchmarkSummarySchema] = {}
_summary_lang_schemas: OrderedDict[
    tuple[str, tuple[str, ...]], BenchmarkSummarySchema
] = OrderedDict()
_SUMMARY_LANG_MAX = 256
_task_score_schemas: dict[str, TaskScoresSchema] = {}
_model_score_schemas: dict[str, ModelScoresSchema] = {}
_per_language_schemas: dict[str, BenchmarkPerLanguageSchema] = {}

# Parallel serialised caches — same keys as the schema caches above. Routes
# that just stream bytes back to the client read from these.
_summary_bytes: dict[str, Serialized] = {}
_summary_lang_bytes: OrderedDict[tuple[str, tuple[str, ...]], Serialized] = (
    OrderedDict()
)
_task_score_bytes: dict[str, Serialized] = {}
_model_score_bytes: dict[str, Serialized] = {}
_per_language_bytes: dict[str, Serialized] = {}


# Per-key asyncio.Lock so two concurrent cold requests don't both run a build.
_summary_locks: dict[str, asyncio.Lock] = {}
_summary_lang_locks: dict[tuple[str, tuple[str, ...]], asyncio.Lock] = {}
_per_language_locks: dict[str, asyncio.Lock] = {}
_task_score_locks: dict[str, asyncio.Lock] = {}
_model_score_locks: dict[str, asyncio.Lock] = {}


def _lock_for(locks: dict, key) -> asyncio.Lock:
    lock = locks.get(key)
    if lock is None:
        lock = locks.setdefault(key, asyncio.Lock())
    return lock


async def get_summary(
    name: str, languages: tuple[str, ...] = ()
) -> BenchmarkSummarySchema:
    """Return the cached summary for ``name``, optionally scoped to ``languages``."""
    if not languages:
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

    key = (name, languages)
    cached = _summary_lang_schemas.get(key)
    if cached is not None:
        _summary_lang_schemas.move_to_end(key)
        return cached
    async with _lock_for(_summary_lang_locks, key):
        cached = _summary_lang_schemas.get(key)
        if cached is not None:
            _summary_lang_schemas.move_to_end(key)
            return cached
        logger.info("Building summary for %s (langs=%s)", name, languages)
        schema = await build_benchmark_summary(name, get_cache(), languages=languages)
        _summary_lang_schemas[key] = schema
        if len(_summary_lang_schemas) > _SUMMARY_LANG_MAX:
            evicted, _ = _summary_lang_schemas.popitem(last=False)
            _summary_lang_locks.pop(evicted, None)
            _summary_lang_bytes.pop(evicted, None)
        return schema


async def get_summary_bytes(name: str, languages: tuple[str, ...] = ()) -> Serialized:
    """Return JSON bytes + ETag for ``get_summary(name, languages)``."""
    if not languages:
        cached = _summary_bytes.get(name)
        if cached is not None:
            return cached
        schema = await get_summary(name)
        cached = _serialize(schema)
        _summary_bytes[name] = cached
        return cached
    key = (name, languages)
    cached = _summary_lang_bytes.get(key)
    if cached is not None:
        _summary_lang_bytes.move_to_end(key)
        return cached
    schema = await get_summary(name, languages)
    cached = _serialize(schema)
    _summary_lang_bytes[key] = cached
    if len(_summary_lang_bytes) > _SUMMARY_LANG_MAX:
        _summary_lang_bytes.popitem(last=False)
    return cached


async def get_per_language(name: str) -> BenchmarkPerLanguageSchema:
    """Return the cached per-language schema for ``name``."""
    cached = _per_language_schemas.get(name)
    if cached is not None:
        return cached
    async with _lock_for(_per_language_locks, name):
        cached = _per_language_schemas.get(name)
        if cached is not None:
            return cached
        from mteb.api.aggregators import build_benchmark_per_language

        logger.info("Building per-language scores for %s", name)
        schema = await build_benchmark_per_language(name)
        _per_language_schemas[name] = schema
        return schema


async def get_per_language_bytes(name: str) -> Serialized:
    """JSON bytes + ETag for ``get_per_language(name)``."""
    cached = _per_language_bytes.get(name)
    if cached is not None:
        return cached
    schema = await get_per_language(name)
    cached = _serialize(schema)
    _per_language_bytes[name] = cached
    return cached


async def get_task_scores(name: str) -> TaskScoresSchema:
    """Return the cached task-scores schema for ``name``."""
    cached = _task_score_schemas.get(name)
    if cached is not None:
        return cached
    async with _lock_for(_task_score_locks, name):
        cached = _task_score_schemas.get(name)
        if cached is not None:
            return cached
        from mteb.api.aggregators import build_task_scores

        schema = await asyncio.to_thread(build_task_scores, name, get_cache())
        _task_score_schemas[name] = schema
        return schema


async def get_task_scores_bytes(name: str) -> Serialized:
    """JSON bytes + ETag for ``get_task_scores(name)``."""
    cached = _task_score_bytes.get(name)
    if cached is not None:
        return cached
    schema = await get_task_scores(name)
    cached = _serialize(schema)
    _task_score_bytes[name] = cached
    return cached


async def get_model_scores(name: str) -> ModelScoresSchema:
    """Return the cached model-scores schema for ``name``."""
    cached = _model_score_schemas.get(name)
    if cached is not None:
        return cached
    async with _lock_for(_model_score_locks, name):
        cached = _model_score_schemas.get(name)
        if cached is not None:
            return cached
        from mteb.api.aggregators import build_model_scores

        schema = await build_model_scores(name)
        _model_score_schemas[name] = schema
        return schema


async def get_model_scores_bytes(name: str) -> Serialized:
    """JSON bytes + ETag for ``get_model_scores(name)``."""
    cached = _model_score_bytes.get(name)
    if cached is not None:
        return cached
    schema = await get_model_scores(name)
    cached = _serialize(schema)
    _model_score_bytes[name] = cached
    return cached


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
    """Pre-build the unfiltered list schemas + their serialised bytes."""
    from mteb.api.routes import (
        _benchmark_schemas_bytes,
        _filtered_model_schemas_bytes,
        _filtered_task_schemas_bytes,
        _menu_schemas_bytes,
    )

    _menu_schemas_bytes()
    _benchmark_schemas_bytes()
    _filtered_task_schemas_bytes(None, None, None, None, None)
    _filtered_model_schemas_bytes(None, None, None, None, None, None, None, False)


def warmup_blocking() -> None:
    """Populate shared caches synchronously before accepting requests.

    Phases 1-3 are independent (network I/O / CPU / pydantic-core); they run
    on a small thread pool so wall time is max() instead of sum(). Phase 4
    (list schemas) depends on phases 1-3 because it reads them through the
    benchmark / task / model caches, so it runs serially afterwards.
    """
    from concurrent.futures import ThreadPoolExecutor

    from mteb.api.adapters import prewarm_schema_caches

    try:
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="warmup") as ex:
            futures = [
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
    if not preload_full():
        return

    def _run() -> None:
        from mteb.api.aggregators import _flat_leaderboard_benchmarks

        all_names = [b.name for b in _flat_leaderboard_benchmarks()]

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
