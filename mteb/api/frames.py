"""Long-results polars frames — process-wide, loaded once.

Pulled out of :mod:`mteb.api.cache` so :mod:`mteb.api.aggregators` (which only
needs the long frames, not the bytes caches) can depend on this module without
pulling in the whole cache layer. Sits below ``cache`` and ``routes`` in the
import graph.
"""

from __future__ import annotations

import functools
import logging
from typing import cast

import polars as pl
from datasets import load_dataset

import mteb
from mteb.api.settings import get_settings
from mteb.cache.result_cache import ResultCache
from mteb.results.benchmark_results import BenchmarkResults

logger = logging.getLogger(__name__)


@functools.cache
def get_cache() -> ResultCache:
    """Return the process-wide :class:`ResultCache`."""
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
        combined.lazy()
        .drop_nulls("score")
        .group_by(["model_name", "task_name", "subset"])
        .agg(pl.col("score").max())
        .collect(engine="streaming")
    )


_DEFAULT_CONFIG = "default"


def _load_default_from_hub(repo_id: str) -> pl.DataFrame | None:
    """Fetch the ``default`` HF config (all-results dump) via direct parquet read.

    Reads parquet shards with polars rather than ``datasets.load_dataset`` so a
    stale README ``dataset_info`` block (e.g. when a new column lands in the
    parquet but the YAML wasn't refreshed) doesn't fail the cast.

    Catches only the exception families either path is expected to raise on a
    legitimate miss: ``OSError`` (network / FS / HTTP, includes
    ``ConnectionError`` + ``FileNotFoundError``), ``ValueError`` (unknown
    config, malformed args), ``KeyError`` (missing parquet/metadata key), and
    polars' own ``PolarsError`` base. ``Exception`` would hide programmer bugs
    like ``TypeError`` / ``AttributeError``.
    """
    expected = (OSError, ValueError, KeyError, pl.exceptions.PolarsError)
    try:
        return pl.read_parquet(f"hf://datasets/{repo_id}/data/train-*.parquet")
    except expected as exc:
        logger.warning(
            "Hub load failed for %s: %s: %s", repo_id, type(exc).__name__, exc
        )
        try:
            return cast(
                "pl.DataFrame",
                load_dataset(repo_id, name=_DEFAULT_CONFIG, split="train").to_polars(),
            )
        except expected as exc2:
            logger.warning(
                "Hub fallback also failed for %s/%s: %s: %s",
                repo_id,
                _DEFAULT_CONFIG,
                type(exc2).__name__,
                exc2,
            )
            return None


@functools.cache
def _load_per_benchmark_frames() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return ``(per_benchmark_frames, unified_frame)``, loaded once.

    Callers MUST NOT mutate the returned frames — they are shared across the
    whole process. Polars frames are practically immutable from the Python
    API, but ``.with_columns`` etc. should be called on `.lazy()` clones.
    """
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
