"""Process-wide caches for the API.

``ResultCache`` is cheap to construct but :meth:`ResultCache.load_results`
downloads / scans the on-disk results tree (slow first-call, ~tens of seconds).
We keep one ``ResultCache`` instance per process and memoise the per-benchmark
summary so repeat requests cost nothing.

Caches hold **pydantic schemas** (one per endpoint) instead of pre-serialised
JSON bytes. Pydantic v2's ``model_dump_json`` runs in Rust (``pydantic-core``)
and ends up I/O-bound on a ~500 KB payload, so serialising once per warm
request was measured to add only single-digit ms — well within the headroom
freed by skipping the bytes cache layer.

Cold builds are serialised through a per-benchmark threading lock so that ten
concurrent requests for the same benchmark only run the polars pipeline once
— previously they each kicked off their own build, saturating the threadpool.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import os
import threading  # still needed for warmup daemon thread
from typing import TYPE_CHECKING

import polars as pl

from mteb.api.aggregators import build_benchmark_summary
from mteb.api.settings import preload_full

if TYPE_CHECKING:
    from collections.abc import Callable

    from mteb.api.schemas import (
        BenchmarkSummarySchema,
        ModelScoresSchema,
        TaskScoresSchema,
    )
    from mteb.cache.result_cache import ResultCache

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def get_cache() -> ResultCache:
    """Return the process-wide :class:`ResultCache` (one instance per API process)."""
    from mteb.cache.result_cache import ResultCache

    return ResultCache()


_DEFAULT_CACHE_REPO_ID = "mteb/results"


def _cache_repo_id() -> str:
    """HF dataset id the Gradio leaderboard pulls its parquet from.

    Override with ``MTEB_API_CACHE_REPO`` (set to ``""`` to disable hub load).
    """
    val = os.environ.get("MTEB_API_CACHE_REPO")
    if val is None:
        return _DEFAULT_CACHE_REPO_ID
    return val


@dataclasses.dataclass(frozen=True)
class _LoadedFrames:
    """Pair of views over the same parquet load: per-benchmark + unified."""

    per_benchmark: dict[str, pl.DataFrame]
    # Schema: (model_name, task_name, subset, score) deduped via max(score)
    # across splits/languages/benchmarks. Built once during the same load so
    # callers that want the unified view never have to recompute.
    all_results: pl.DataFrame


def _build_unified_frame(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    non_empty = [df for df in frames.values() if not df.is_empty()]
    if not non_empty:
        return pl.DataFrame(
            schema={
                "model_name": pl.Utf8,
                "task_name": pl.Utf8,
                "subset": pl.Utf8,
                "score": pl.Float64,
            }
        )
    combined = pl.concat(non_empty, how="vertical_relaxed")
    return (
        combined.drop_nulls("score")
        .group_by(["model_name", "task_name", "subset"])
        .agg(pl.col("score").max())
    )


@functools.lru_cache(maxsize=1)
def _load_per_benchmark_frames() -> _LoadedFrames:
    """Mirror of the Gradio leaderboard's bootstrap loader.

    Priority order (same as :func:`mteb.leaderboard.app.get_leaderboard_app`):

    1. ``hf://datasets/{MTEB_API_CACHE_REPO}/leaderboard/benchmark_results.parquet``
       (default ``mteb/results``)
    2. Local parquet at ``{ResultCache.cache_path}/leaderboard/benchmark_results.parquet``
    3. Cold rebuild via ``cache._load_from_cache(rebuild=True)`` + persist
       the parquet for next time.

    Returns a :class:`_LoadedFrames` with both views computed once:

    * ``per_benchmark`` — ``{benchmark_name: long polars frame}`` for the
      Gradio-compatible per-benchmark summary path.
    * ``all_results`` — one ``(model_name, task_name, subset, score)`` frame
      deduped across benchmarks, ready for direct task-scope filtering.
    """
    import mteb
    from mteb.results.benchmark_results import BenchmarkResults

    cache = get_cache()
    parquet_path = cache.leaderboard_parquet_path

    def _try(
        source: str, loader: Callable[[], dict[str, pl.DataFrame]]
    ) -> dict[str, pl.DataFrame] | None:
        try:
            loaded = loader()
        except Exception as exc:
            logger.warning("Leaderboard cache load failed for %s: %s", source, exc)
            return None
        if not loaded:
            logger.info("Leaderboard cache from %s is empty/outdated", source)
            return None
        logger.info("Loaded %d benchmark frames from %s", len(loaded), source)
        return loaded

    loaded: dict[str, pl.DataFrame] | None = None
    repo_id = _cache_repo_id()
    if repo_id:
        loaded = _try(
            f"hub '{repo_id}'",
            lambda: BenchmarkResults.load_leaderboard_cache(repo_id, from_hub=True),
        )

    if loaded is None and parquet_path.exists():
        loaded = _try(
            f"local parquet {parquet_path}",
            lambda: BenchmarkResults.load_leaderboard_cache(parquet_path),
        )

    if loaded is None:
        logger.info(
            "Building leaderboard parquet cache from scratch at %s", parquet_path
        )
        all_results = cache._load_from_cache(rebuild=True)
        benchmarks = mteb.get_benchmarks(display_on_leaderboard=True)
        loaded = {b.name: all_results.to_results_df(b.tasks) for b in benchmarks}
        BenchmarkResults.save_leaderboard_cache(loaded, parquet_path)

    unified = _build_unified_frame(loaded)
    logger.info(
        "Built unified results frame: %d rows / %d unique tasks",
        unified.height,
        unified.select(pl.col("task_name").n_unique()).item() if unified.height else 0,
    )
    return _LoadedFrames(per_benchmark=loaded, all_results=unified)


def get_all_benchmark_frames() -> dict[str, pl.DataFrame]:
    """Per-benchmark long polars frames keyed by benchmark name (Gradio-compatible)."""
    return _load_per_benchmark_frames().per_benchmark


def get_all_results_df() -> pl.DataFrame:
    """One unified ``(model_name, task_name, subset, score)`` frame for every task."""
    return _load_per_benchmark_frames().all_results


# Plain-dict schema caches. We rely on the GIL for dict atomicity and on
# pydantic schemas being immutable-by-convention. There's no explicit
# serialisation around the cold build: two requests racing the same
# uncached benchmark may both kick off ``build_benchmark_summary`` and the
# second to finish "wins" the cache slot — wasteful but correctness-safe.
# In practice this happens only during a cold boot before
# ``warmup_in_background`` lands a result, and the warmup itself sequences
# them via ``asyncio.gather`` on a single loop.
_summary_schemas: dict[str, BenchmarkSummarySchema] = {}
_task_score_schemas: dict[str, TaskScoresSchema] = {}
_model_score_schemas: dict[str, ModelScoresSchema] = {}


async def get_summary(name: str) -> BenchmarkSummarySchema:
    """Return the cached :class:`BenchmarkSummarySchema` for ``name`` (build on miss)."""
    cached = _summary_schemas.get(name)
    if cached is not None:
        return cached
    logger.info("Building summary for %s", name)
    schema = await build_benchmark_summary(name, get_cache())
    _summary_schemas[name] = schema
    return schema


async def get_task_scores(name: str) -> TaskScoresSchema:
    """Return the cached :class:`TaskScoresSchema` for ``name`` (build on miss)."""
    cached = _task_score_schemas.get(name)
    if cached is not None:
        return cached
    from mteb.api.aggregators import build_task_scores

    # ``build_task_scores`` is sync polars work — hop to the default executor
    # so we don't block the event loop while it runs.
    schema = await asyncio.to_thread(build_task_scores, name, get_cache())
    _task_score_schemas[name] = schema
    return schema


async def get_model_scores(name: str) -> ModelScoresSchema:
    """Return the cached :class:`ModelScoresSchema` for ``name`` (build on miss)."""
    cached = _model_score_schemas.get(name)
    if cached is not None:
        return cached
    from mteb.api.aggregators import build_model_scores

    schema = await build_model_scores(name)
    _model_score_schemas[name] = schema
    return schema


def _prewarm_training_datasets() -> None:
    """Populate ``_training_datasets_cached`` for every registered model.

    The first ``build_benchmark_summary`` call pays ~2.5s on
    ``_collect_similar_tasks`` (a per-model difflib walk) because every model
    seen for the first time forces a similar-task graph traversal.
    ``_training_datasets_cached`` memoises that result and is process-wide, so
    once it's filled every subsequent benchmark summary builds in O(polars
    aggregation) only. Running this in parallel at startup shifts the cost
    from request time to boot.
    """
    from concurrent.futures import ThreadPoolExecutor

    from mteb.benchmarks._create_table import _training_datasets_cached
    from mteb.models.model_implementations import MODEL_REGISTRY

    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="warm-td") as ex:
        list(ex.map(_training_datasets_cached, MODEL_REGISTRY))


def _prewarm_list_schemas() -> None:
    """Pre-build the unfiltered ``/tasks`` / ``/models`` / ``/benchmarks`` schema lists.

    These are bounded-size pydantic lists (1000+ entries) cached at the route
    layer. Building them once at startup turns the first hit on each list
    endpoint into a cached lookup + a single pydantic serialisation pass
    instead of also paying the per-task / per-model schema construction.
    """
    from mteb.api.routes import (
        _benchmark_schemas,
        _filtered_model_schemas,
        _filtered_task_schemas,
        _menu_schemas,
    )

    _menu_schemas()
    _benchmark_schemas()
    _filtered_task_schemas(None, None, None, None, None, None)
    _filtered_model_schemas(None, None, None, None, None, None, None, False, None)


def warmup_in_background() -> None:
    """Warm shared API caches at process startup.

    Runs on a daemon thread so the HTTP listener comes up immediately. The
    *light* path is always on (parquet frames + per-model training-datasets +
    pydantic schema caches + unfiltered list JSONs) because every dollar
    spent here turns a per-request cold path into a hot one. The *heavy*
    path — actually building every benchmark summary — is gated by
    ``MTEB_API_PRELOAD=1`` because it dominates cold boot (~2-4s per
    benchmark × 56 benchmarks even with parallelism).

    Per-benchmark failures are logged but never crash startup; mteb's
    registries are static after import, so a thrown exception almost always
    means a benchmark or a model is mis-configured rather than a transient
    error.
    """

    def _run() -> None:
        from mteb.api.adapters import prewarm_schema_caches

        # Light path (always on): load shared frames + memoised metadata so
        # the first call to /benchmarks/X/summary doesn't burn ~2.5s in
        # _collect_similar_tasks and the first /tasks call doesn't burn
        # ~400ms instantiating + serialising 1000 task schemas.
        try:
            get_all_benchmark_frames()
            _prewarm_training_datasets()
            prewarm_schema_caches()
            _prewarm_list_schemas()
        except Exception as exc:
            logger.warning("light warmup failed: %s", exc)
            return

        if not preload_full():
            return

        from mteb.api.aggregators import _flat_leaderboard_benchmarks

        all_names = [b.name for b in _flat_leaderboard_benchmarks()]

        async def _build_one(name: str) -> None:
            try:
                await get_summary(name)
            except Exception as exc:
                logger.warning("warmup: %s failed (%s)", name, exc)

        async def _build_all() -> None:
            # ``asyncio.gather`` fans out every benchmark summary onto the
            # default thread executor via ``asyncio.to_thread`` (inside the
            # cache + aggregators). Default pool size is ``min(32, cpu+4)``,
            # which covers 56 benchmarks comfortably.
            await asyncio.gather(*(_build_one(n) for n in all_names))

        asyncio.run(_build_all())

    threading.Thread(target=_run, name="mteb-api-warmup", daemon=True).start()
