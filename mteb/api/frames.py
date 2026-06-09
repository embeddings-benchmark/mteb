"""Long-results polars frames — process-wide, loaded once.

Pulled out of :mod:`mteb.api.cache` so :mod:`mteb.api.aggregators` (which only
needs the long frames, not the bytes caches) can depend on this module without
pulling in the whole cache layer. Sits below ``cache`` and ``routes`` in the
import graph.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import quote, unquote

import polars as pl
from datasets import load_dataset

import mteb
from mteb.api.settings import get_settings
from mteb.cache.result_cache import ResultCache
from mteb.results.benchmark_results import BenchmarkResults

if TYPE_CHECKING:
    from collections.abc import Iterable

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
    except expected as exc:
        logger.warning(
            "Hub fallback also failed for %s/%s: %s: %s",
            repo_id,
            _DEFAULT_CONFIG,
            type(exc).__name__,
            exc,
        )
        return None


# Disk-cache schema version. Bump when the on-disk layout changes so old
# caches are invalidated automatically (e.g. if we add columns to the split
# parquets or change the manifest shape).
_DISK_CACHE_VERSION = 1
_UNIFIED_FILE = "_unified.parquet"
_MANIFEST_FILE = "manifest.json"


def _disk_cache_dir() -> Path:
    """Resolve the disk-cache directory; honours XDG_CACHE_HOME, else ``~/.cache``.

    Files: ``manifest.json`` + ``_unified.parquet`` + one ``<urlsafe>.parquet``
    per benchmark with rows.
    """
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "mteb" / "leaderboard"


def _hf_dataset_sha(repo_id: str) -> str | None:
    """Cheap (~100ms) commit-SHA lookup for the HF dataset.

    Used as the disk-cache invalidation key. Returns ``None`` on any failure
    (offline, auth, timeout) — callers treat this as "trust the on-disk cache
    if present" rather than rebuilding offline.
    """
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(repo_id, timeout=5)
        return cast("str", info.sha)
    except Exception as exc:
        logger.info(
            "Could not fetch HF sha for %s (%s); will trust on-disk cache if present.",
            repo_id,
            type(exc).__name__,
        )
        return None


def _read_disk_cache(  # noqa: PLR0911 — multiple validation branches each surface a distinct rebuild reason
    repo_id: str, current_sha: str | None
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame] | None:
    """Return ``(per_benchmark_frames, unified)`` from disk cache, or ``None``.

    Returns ``None`` when:
    * the manifest is missing or malformed,
    * the cache version doesn't match,
    * the cached repo_id differs,
    * an HF sha was fetched and doesn't match the manifest,
    * the unified parquet is missing.

    Trusts the cache when the HF sha probe failed (``current_sha is None``)
    so an offline restart still boots fast.
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
    if manifest.get("version") != _DISK_CACHE_VERSION:
        logger.info("disk cache version mismatch; rebuilding")
        return None
    if manifest.get("repo_id") != repo_id:
        logger.info("disk cache repo_id mismatch; rebuilding")
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
    """Persist the split + unified frames so the next process restart skips the cold work.

    Saves the ~20s HF download and ~10s split. Writes the manifest LAST so a
    crashed write leaves the cache invalid (next read sees no manifest,
    rebuilds cleanly).
    """
    cache_dir = _disk_cache_dir()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Remove stale shards before writing new ones (cheap; covers benchmark
        # renames and reduces clutter when the registry shrinks).
        for p in cache_dir.glob("*.parquet"):
            p.unlink(missing_ok=True)
        manifest_path = cache_dir / _MANIFEST_FILE
        manifest_path.unlink(missing_ok=True)

        unified.write_parquet(cache_dir / _UNIFIED_FILE)
        for bench_name, df in loaded.items():
            # Skip the placeholder empty frames added by the setdefault loop —
            # they're cheap to recreate on read and waste disk.
            if df.height == 0:
                continue
            safe_name = quote(bench_name, safe="")
            df.write_parquet(cache_dir / f"{safe_name}.parquet")
        # Manifest is the last write — anything before is "tentative".
        manifest_path.write_text(
            json.dumps(
                {
                    "version": _DISK_CACHE_VERSION,
                    "repo_id": repo_id,
                    "sha": sha,
                }
            )
        )
    except OSError as exc:
        # Failing to write is non-fatal — the in-memory result is still valid.
        logger.warning("disk cache write failed: %s", exc)


def _fill_empty_placeholders(
    loaded: dict[str, pl.DataFrame],
    benches: Iterable,
    schema: dict[str, pl.DataType],
) -> None:
    """Add empty-frame placeholders for benchmarks with no rows.

    Re-applied after both fresh build and disk-cache load so newly-added
    benchmarks (registry adds since cache was written) get the same
    placeholder treatment downstream consumers expect.
    """
    for bench in benches:
        loaded.setdefault(bench.name, pl.DataFrame(schema=schema))


@functools.cache
def _load_per_benchmark_frames() -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    """Return ``(per_benchmark_frames, unified_frame)``, loaded once.

    Callers MUST NOT mutate the returned frames — they are shared across the
    whole process. Polars frames are practically immutable from the Python
    API, but ``.with_columns`` etc. should be called on `.lazy()` clones.

    Disk-cache layer: when ``DISK_CACHE`` is enabled (default), persists the
    split + unified frames to ``~/.cache/mteb/per_benchmark_frames/`` and
    skips the ~20s HF download + ~10s split on subsequent process restarts.
    Invalidated via the HF dataset commit SHA.
    """
    settings = get_settings()
    repo_id = settings.cache_repo
    disk_cache_enabled = bool(settings.disk_cache and repo_id)

    # Fetch sha once: used both to validate any existing cache and to record
    # in the new manifest if we end up rebuilding. ``None`` means offline —
    # ``_read_disk_cache`` treats that as "trust the on-disk copy".
    current_sha = _hf_dataset_sha(repo_id) if disk_cache_enabled else None

    if disk_cache_enabled:
        cached = _read_disk_cache(repo_id, current_sha)
        if cached is not None:
            loaded, unified = cached
            _fill_empty_placeholders(
                loaded, mteb.get_benchmarks(), dict(unified.schema)
            )
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
        all_results = cache._rebuild_from_full_repository()
        combined = all_results._to_results_df()

    loaded = BenchmarkResults.split_leaderboard_frame(combined)
    _fill_empty_placeholders(loaded, mteb.get_benchmarks(), dict(combined.schema))

    unified = _dedupe_unified(combined)
    logger.info(
        "Built unified results frame: %d rows / %d unique tasks",
        unified.height,
        unified.select(pl.col("task_name").n_unique()).item() if unified.height else 0,
    )

    if disk_cache_enabled:
        _write_disk_cache(repo_id, current_sha, loaded, unified)

    return loaded, unified
