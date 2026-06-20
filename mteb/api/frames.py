"""Long-results polars frames — process-wide, loaded once."""

from __future__ import annotations

import functools
import json
import logging
from typing import TYPE_CHECKING, cast
from urllib.parse import quote, unquote

import polars as pl
from datasets import load_dataset
from huggingface_hub import HfApi

import mteb
from mteb.api._errors import FRAME_LOAD_ERRORS
from mteb.api.settings import get_settings
from mteb.cache.result_cache import ResultCache
from mteb.results.benchmark_results import BenchmarkResults

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@functools.cache
def get_cache() -> ResultCache:
    """Return the process-wide :class:`ResultCache`."""
    return ResultCache()


# Bump when the on-disk parquet schema changes so old caches are rebuilt.
_CACHE_SCHEMA_VERSION = 2

_UNIFIED_SCHEMA = {
    "model_name": pl.Utf8,
    "task_name": pl.Utf8,
    "split": pl.Utf8,
    "subset": pl.Utf8,
    "score": pl.Float64,
}


def _dedupe_unified(combined: pl.DataFrame) -> pl.DataFrame:
    """One row per (model, task, split, subset) with max score across duplicate revisions."""
    if combined.is_empty():
        return pl.DataFrame(schema=_UNIFIED_SCHEMA)
    return (
        combined.lazy()
        .drop_nulls("score")
        .group_by(["model_name", "task_name", "split", "subset"])
        .agg(pl.col("score").max())
        .collect(engine="streaming")
    )


_DEFAULT_CONFIG = "default"


def _load_default_from_hub(repo_id: str) -> pl.DataFrame | None:
    """Fetch the ``default`` HF config (all-results dump) via direct parquet read.

    Reads with polars rather than ``datasets.load_dataset`` so a stale README
    ``dataset_info`` block doesn't fail the cast. Falls back to load_dataset.
    """
    try:
        return pl.read_parquet(f"hf://datasets/{repo_id}/data/train-*.parquet")
    except FRAME_LOAD_ERRORS as exc:
        logger.warning(
            "Hub load failed for %s: %s: %s", repo_id, type(exc).__name__, exc
        )
    # Fallback is slower; logged so we notice if direct-parquet keeps missing.
    logger.info(
        "Falling back to datasets.load_dataset for %s/%s", repo_id, _DEFAULT_CONFIG
    )
    try:
        return cast(
            "pl.DataFrame",
            load_dataset(repo_id, name=_DEFAULT_CONFIG, split="train").to_polars(),
        )
    except FRAME_LOAD_ERRORS as exc:
        logger.warning(
            "Hub fallback also failed for %s/%s: %s: %s",
            repo_id,
            _DEFAULT_CONFIG,
            type(exc).__name__,
            exc,
        )
        return None


_UNIFIED_FILE = "_unified.parquet"
_MANIFEST_FILE = "manifest.json"


def _disk_cache_dir() -> Path:
    """Disk-cache dir under the shared :class:`ResultCache` root.

    Files: ``manifest.json`` + ``_unified.parquet`` + one ``<urlsafe>.parquet``
    per benchmark with rows.
    """
    return get_cache().cache_path / "leaderboard"


def _hf_dataset_sha(repo_id: str) -> str | None:
    """Commit-SHA lookup for the HF dataset; ``None`` on failure (offline/auth)."""
    try:
        info = HfApi().dataset_info(repo_id, timeout=5)
        return cast("str", info.sha)
    except Exception as exc:
        logger.info(
            "Could not fetch HF sha for %s (%s); will trust on-disk cache if present.",
            repo_id,
            type(exc).__name__,
        )
        return None


def _read_disk_cache(  # noqa: PLR0911
    repo_id: str, current_sha: str | None
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame] | None:
    """Return ``(per_benchmark_frames, unified)`` from disk cache, or ``None`` on miss.

    Trusts the cache when ``current_sha is None`` so offline restarts boot fast.
    """
    cache_dir = _disk_cache_dir()
    manifest_path = cache_dir / _MANIFEST_FILE
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("disk cache manifest unreadable (%s); rebuilding", exc)
        return None
    if manifest.get("repo_id") != repo_id:
        logger.info("disk cache repo_id mismatch; rebuilding")
        return None
    if manifest.get("schema_version") != _CACHE_SCHEMA_VERSION:
        logger.info("disk cache schema version mismatch; rebuilding")
        return None
    if current_sha is not None and manifest.get("sha") != current_sha:
        logger.info("disk cache SHA mismatch (HF dataset updated); rebuilding")
        return None
    unified_path = cache_dir / _UNIFIED_FILE
    if not unified_path.exists():
        logger.warning("disk cache unified frame missing; rebuilding")
        return None

    try:
        unified = pl.read_parquet(unified_path)
        loaded: dict[str, pl.DataFrame] = {}
        for parquet_path in cache_dir.glob("*.parquet"):
            if parquet_path.name == _UNIFIED_FILE:
                continue
            bench_name = unquote(parquet_path.stem)
            loaded[bench_name] = pl.read_parquet(parquet_path)
    except (OSError, pl.exceptions.PolarsError) as exc:
        logger.warning("disk cache read failed (%s); rebuilding", exc)
        return None

    return loaded, unified


def _write_disk_cache(
    repo_id: str,
    sha: str | None,
    loaded: dict[str, pl.DataFrame],
    unified: pl.DataFrame,
) -> None:
    """Persist split + unified frames so the next restart skips the cold work.

    Each shard is written via ``.tmp`` + ``os.replace`` so readers and abrupt
    exits never see a torn state.
    """
    cache_dir = _disk_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Skip placeholder empty frames (cheap to recreate on read).
        new_shards: list[tuple[Path, pl.DataFrame]] = [
            (cache_dir / _UNIFIED_FILE, unified)
        ]
        for bench_name, df in loaded.items():
            if df.height == 0:
                continue
            safe_name = quote(bench_name, safe="")
            new_shards.append((cache_dir / f"{safe_name}.parquet", df))

        for final_path, df in new_shards:
            tmp_path = final_path.with_name(final_path.name + ".tmp")
            df.write_parquet(tmp_path)
            tmp_path.replace(final_path)

        manifest_path = cache_dir / _MANIFEST_FILE
        manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
        manifest_tmp.write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "sha": sha,
                    "schema_version": _CACHE_SCHEMA_VERSION,
                }
            )
        )
        manifest_tmp.replace(manifest_path)

        keep = {p.name for p, _ in new_shards}
        for p in cache_dir.glob("*.parquet"):
            if p.name not in keep:
                p.unlink(missing_ok=True)
    except OSError as exc:
        # Failing to write is non-fatal — the in-memory result is still valid.
        logger.warning("disk cache write failed: %s", exc)


@functools.cache
def _load_per_benchmark_frames() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return ``(per_benchmark_frames, unified_frame)``, loaded once.

    When ``DISK_CACHE`` is enabled, persists frames under ``ResultCache`` root
    and skips the HF download + split on subsequent restarts. Invalidated via
    the HF dataset commit SHA.
    """
    settings = get_settings()
    repo_id = settings.cache_repo
    disk_cache_repo: str | None = repo_id if settings.disk_cache and repo_id else None

    current_sha = _hf_dataset_sha(disk_cache_repo) if disk_cache_repo else None

    if disk_cache_repo:
        cached = _read_disk_cache(disk_cache_repo, current_sha)
        if cached is not None:
            loaded, unified = cached
            empty = pl.DataFrame(schema=dict(unified.schema))
            for bench in mteb.get_benchmarks():
                loaded.setdefault(bench.name, empty)
            logger.info(
                "Loaded per-benchmark frames from disk cache "
                "(%d benchmarks, unified=%d rows)",
                sum(1 for df in loaded.values() if df.height > 0),
                unified.height,
            )
            return loaded, unified

    cache = get_cache()
    combined: pl.DataFrame | None = None

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
    empty = pl.DataFrame(schema=dict(combined.schema))
    for bench in mteb.get_benchmarks():
        loaded.setdefault(bench.name, empty)

    unified = _dedupe_unified(combined)
    logger.info(
        "Built unified results frame: %d rows / %d unique tasks",
        unified.height,
        unified.select(pl.col("task_name").n_unique()).item() if unified.height else 0,
    )

    if disk_cache_repo:
        _write_disk_cache(disk_cache_repo, current_sha, loaded, unified)

    return loaded, unified
