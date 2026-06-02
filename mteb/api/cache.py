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
import functools
import logging
import threading
from typing import TYPE_CHECKING

import polars as pl

from mteb.api.aggregators import build_benchmark_summary
from mteb.api.settings import cache_repo, preload_full

if TYPE_CHECKING:
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


# ``_load_per_benchmark_frames`` returns ``(per_benchmark, all_results)``:
#   * per_benchmark — long polars frame per benchmark name
#   * all_results  — deduped (model_name, task_name, subset, score) frame
#     covering every task; built once during the same load so callers that
#     want the unified view never have to recompute.


_UNIFIED_SCHEMA = {
    "model_name": pl.Utf8,
    "task_name": pl.Utf8,
    "subset": pl.Utf8,
    "score": pl.Float64,
}


def _dedupe_unified(combined: pl.DataFrame) -> pl.DataFrame:
    """Reduce the combined frame to one row per (model, task, subset).

    Multiple splits/languages/benchmark tags collapse into ``max(score)``
    so the unified view is a single point per task — what task-scope
    filtering (`/tasks/{name}` etc.) needs.
    """
    if combined.is_empty():
        return pl.DataFrame(schema=_UNIFIED_SCHEMA)
    return (
        combined.drop_nulls("score")
        .group_by(["model_name", "task_name", "subset"])
        .agg(pl.col("score").max())
    )


_DEFAULT_CONFIG = "default"


def _load_default_from_hub(repo_id: str) -> pl.DataFrame | None:
    """Fetch only the ``default`` HF config — the all-results dump.

    Per-benchmark views are derived locally via
    :meth:`BenchmarkResults.split_leaderboard_frame` rather than
    fetching ``N`` configs from the hub one at a time, which was the
    dominant cost at startup (one HTTP round-trip per benchmark).
    """
    from datasets import load_dataset

    try:
        return load_dataset(repo_id, name=_DEFAULT_CONFIG, split="train").to_polars()
    except Exception as exc:
        logger.warning("Hub load failed for %s/%s: %s", repo_id, _DEFAULT_CONFIG, exc)
        return None


@functools.lru_cache(maxsize=1)
def _load_per_benchmark_frames() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Bootstrap loader for per-benchmark + unified-results frames.

    Two paths only:

    1. Hub: pull the ``default`` config from ``MTEB_API_CACHE_REPO``
       (default ``mteb/results``) — one round-trip — then split it
       in-memory.
    2. Cold rebuild from the local ``ResultCache`` if the hub returns
       nothing usable.

    Per-benchmark views are derived from the combined frame via
    :meth:`BenchmarkResults.split_leaderboard_frame`, which inner-joins
    each benchmark's ``(task_name, split, subset)`` triples — same
    filter the legacy per-task validator would have applied.
    """
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

    # Per-benchmark dict is derived from the same combined frame —
    # no extra hub round-trips, no per-benchmark cold rebuilds.
    loaded = BenchmarkResults.split_leaderboard_frame(combined)
    # Add explicit empty frames for any leaderboard benchmark missing
    # from the split, so callers using `.get(name)` always see a
    # consistent key set (kept the previous behaviour for callers that
    # check membership before reading).
    for bench in mteb.get_benchmarks():
        loaded.setdefault(bench.name, pl.DataFrame(schema=combined.schema))

    unified = _dedupe_unified(combined)
    logger.info(
        "Built unified results frame: %d rows / %d unique tasks",
        unified.height,
        unified.select(pl.col("task_name").n_unique()).item() if unified.height else 0,
    )
    return loaded, unified


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
            _load_per_benchmark_frames()
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
