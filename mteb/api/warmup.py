"""Startup-time cache warmup orchestration; called from the ASGI lifespan."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import mteb
from mteb.api._errors import FRAME_LOAD_ERRORS, PRELOAD_ERRORS
from mteb.api.adapters import prewarm_schema_caches
from mteb.api.cache import get_per_language_bytes, get_summary_bytes
from mteb.api.frames import _load_per_benchmark_frames
from mteb.api.routes import (
    _benchmark_schemas_bytes,
    _filtered_model_schemas_bytes,
    _filtered_task_schemas_bytes,
    _menu_schemas_bytes,
)
from mteb.api.settings import get_settings
from mteb.benchmarks._create_table import _training_datasets_cached
from mteb.models.model_implementations import MODEL_REGISTRY

if TYPE_CHECKING:
    from concurrent.futures import Future

logger = logging.getLogger(__name__)


def _prewarm_training_datasets() -> None:
    """Populate ``_training_datasets_cached`` for every model.

    Why: first summary build otherwise pays ~2.5s per first-seen model.
    """
    t0 = time.monotonic()
    logger.info("warmup: training-datasets started (%d models)", len(MODEL_REGISTRY))
    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="warm-td") as ex:
        list(ex.map(_training_datasets_cached, MODEL_REGISTRY))
    logger.info("warmup: training-datasets done in %.2fs", time.monotonic() - t0)


def _prewarm_list_schemas() -> None:
    """Pre-build the unfiltered list schemas + serialised bytes (threaded)."""
    t0 = time.monotonic()
    logger.info("warmup: list schemas started")
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
    logger.info("warmup: list schemas done in %.2fs", time.monotonic() - t0)


def _load_per_benchmark_frames_logged() -> None:
    """``_load_per_benchmark_frames`` with timing logs."""
    t0 = time.monotonic()
    logger.info("warmup: per-benchmark frames started")
    _load_per_benchmark_frames()
    logger.info("warmup: per-benchmark frames done in %.2fs", time.monotonic() - t0)


def _prewarm_schema_caches_logged() -> None:
    """``prewarm_schema_caches`` with timing logs."""
    t0 = time.monotonic()
    logger.info("warmup: schema caches started")
    prewarm_schema_caches()
    logger.info("warmup: schema caches done in %.2fs", time.monotonic() - t0)


def warmup_blocking() -> None:
    """Populate shared caches before accepting requests.

    Phases 1-3 (frames / training-datasets / schema caches) run in parallel;
    phase 4 (list schemas) depends on them and runs serially after.
    """
    t0 = time.monotonic()
    logger.info("warmup: blocking phase started")
    try:
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="warmup") as ex:
            futures: list[Future[object]] = [
                ex.submit(_load_per_benchmark_frames_logged),
                ex.submit(_prewarm_training_datasets),
                ex.submit(_prewarm_schema_caches_logged),
            ]
            for f in futures:
                f.result()
        _prewarm_list_schemas()
    except FRAME_LOAD_ERRORS as exc:
        # Non-fatal — routes will rebuild on first request. Narrow set lets
        # programmer errors (TypeError, AttributeError) surface.
        logger.warning(
            "warmup: blocking phase failed after %.2fs (%s: %s)",
            time.monotonic() - t0,
            type(exc).__name__,
            exc,
        )
        return
    logger.info("warmup: blocking phase done in %.2fs", time.monotonic() - t0)


def preload_summaries_in_background() -> asyncio.Task[None] | None:
    """Pre-build every summary + per-language schema on the serving loop.

    Must run on the serving event loop so the per-key ``asyncio.Lock``
    instances are bound to the same loop request handlers use.
    """
    if not get_settings().preload:
        logger.info("warmup: background preload disabled (PRELOAD=0)")
        return None

    concurrency = get_settings().preload_concurrency
    # Include off-menu benchmarks so hidden requests + the model detail page
    # (which iterates all benchmarks) stay cold-path-free.
    all_names = [b.name for b in mteb.get_benchmarks()]

    async def _build_one(name: str) -> None:
        try:
            await get_summary_bytes(name)
        except PRELOAD_ERRORS as exc:
            logger.warning(
                "warmup summary: %s failed (%s: %s)",
                name,
                type(exc).__name__,
                exc,
            )
        try:
            await get_per_language_bytes(name)
        except PRELOAD_ERRORS as exc:
            logger.warning(
                "warmup per-language: %s failed (%s: %s)",
                name,
                type(exc).__name__,
                exc,
            )

    async def _run() -> None:
        t0 = time.monotonic()
        logger.info(
            "warmup: background preload started (%d benchmarks, concurrency=%d)",
            len(all_names),
            concurrency,
        )
        sem = asyncio.Semaphore(concurrency)

        async def _bounded(name: str) -> None:
            async with sem:
                await _build_one(name)

        await asyncio.gather(*(_bounded(n) for n in all_names))
        logger.info(
            "warmup: background preload done in %.2fs (%d benchmarks)",
            time.monotonic() - t0,
            len(all_names),
        )

    return asyncio.create_task(_run(), name="mteb-api-preload")
