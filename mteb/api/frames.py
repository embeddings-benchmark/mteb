"""Long-results polars frames — process-wide, loaded once."""

from __future__ import annotations

import functools
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import cast
from urllib.parse import quote

import polars as pl
from datasets import load_dataset
from huggingface_hub import HfApi

import mteb
from mteb.api._errors import FRAME_LOAD_ERRORS
from mteb.api.settings import get_settings
from mteb.cache.result_cache import ResultCache
from mteb.results.benchmark_results import BenchmarkResults

logger = logging.getLogger(__name__)


@functools.cache
def get_cache() -> ResultCache:
    """Return the process-wide :class:`ResultCache`."""
    return ResultCache()


# Bump when the on-disk parquet schema changes so old caches are rebuilt.
# v3: model_meta's model_type/embed_dim/output_dtypes folded into the
# `experiments` struct column (no more separate `model_meta` column).
# v4: publish all API frames together in one atomic ZIP snapshot.
_CACHE_SCHEMA_VERSION = 4

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


_UNIFIED_FILE = "unified.parquet"
_MANIFEST_FILE = "manifest.json"
_SNAPSHOT_FILE = "api-cache-v4.zip"


def _disk_cache_dir() -> Path:
    """Disk-cache dir under the shared :class:`ResultCache` root.

    The API owns ``api-cache-v4.zip``. Other files in this shared directory,
    including legacy API shards and Gradio's parquet, are left untouched.
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


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate manifest keys instead of silently choosing one."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate disk-cache manifest key")
    return result


def _read_disk_cache(  # noqa: PLR0911
    repo_id: str, current_sha: str | None
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame] | None:
    """Return ``(per_benchmark_frames, unified)`` from disk cache, or ``None`` on miss.

    Trusts the cache when ``current_sha is None`` so offline restarts boot fast.
    """
    try:
        # One handle pins one complete generation, even if its path is replaced.
        with zipfile.ZipFile(_disk_cache_dir() / _SNAPSHOT_FILE) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                return None
            if any(
                info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1
                for info in archive.infolist()
            ):
                return None
            manifest = json.loads(
                archive.read(_MANIFEST_FILE), object_pairs_hook=_unique_json_object
            )
            if not isinstance(manifest, dict):
                return None
            if manifest.get("repo_id") != repo_id:
                return None
            if (
                type(manifest.get("schema_version")) is not int
                or manifest["schema_version"] != _CACHE_SCHEMA_VERSION
            ):
                return None
            if "sha" not in manifest or (
                manifest["sha"] is not None and not isinstance(manifest["sha"], str)
            ):
                return None
            if current_sha is not None and manifest["sha"] != current_sha:
                return None
            entries = manifest.get("benchmarks")
            if not isinstance(entries, dict) or any(
                not isinstance(name, str)
                or not isinstance(entry, str)
                or entry != f"benchmarks/{quote(name, safe='')}.parquet"
                for name, entry in entries.items()
            ):
                return None
            if set(names) != {_MANIFEST_FILE, _UNIFIED_FILE, *entries.values()}:
                return None
            # Reading the complete entry also verifies its ZIP CRC before Polars.
            unified = pl.read_parquet(io.BytesIO(archive.read(_UNIFIED_FILE)))
            loaded = {
                name: pl.read_parquet(io.BytesIO(archive.read(entry)))
                for name, entry in entries.items()
            }
        return loaded, unified
    except FileNotFoundError:
        return None
    except (
        OSError,
        ValueError,
        EOFError,
        KeyError,
        NotImplementedError,
        zipfile.BadZipFile,
        pl.exceptions.PolarsError,
    ) as exc:
        logger.warning("disk cache read failed (%s); rebuilding", exc)
        return None


def _write_disk_cache(
    repo_id: str,
    sha: str | None,
    loaded: dict[str, pl.DataFrame],
    unified: pl.DataFrame,
) -> None:
    """Persist split + unified frames so the next restart skips the cold work.

    All frames are closed in one unique temporary ZIP before publishing it with
    one atomic replacement. Failed writes leave the previous snapshot readable.
    """
    cache_dir = _disk_cache_dir()
    tmp_path: Path | None = None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        entries = {
            name: f"benchmarks/{quote(name, safe='')}.parquet"
            for name, frame in loaded.items()
            if not frame.is_empty()
        }
        with tempfile.NamedTemporaryFile(
            dir=cache_dir, prefix=".api-cache-", suffix=".tmp", delete=False
        ) as temporary:
            tmp_path = Path(temporary.name)
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as archive:
                archive.writestr(
                    _MANIFEST_FILE,
                    json.dumps(
                        {
                            "repo_id": repo_id,
                            "sha": sha,
                            "schema_version": _CACHE_SCHEMA_VERSION,
                            "benchmarks": entries,
                        }
                    ),
                )
                with archive.open(_UNIFIED_FILE, "w", force_zip64=True) as entry:
                    unified.write_parquet(entry)
                for name, filename in entries.items():
                    with archive.open(filename, "w", force_zip64=True) as entry:
                        loaded[name].write_parquet(entry)
        tmp_path.replace(cache_dir / _SNAPSHOT_FILE)
    except (OSError, pl.exceptions.PolarsError, zipfile.BadZipFile) as exc:
        # Failing to write is non-fatal — the in-memory result is still valid.
        logger.warning("disk cache write failed: %s", exc)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("disk cache temporary-file cleanup failed: %s", exc)


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
