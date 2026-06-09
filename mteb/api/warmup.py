"""Startup-time cache warmup orchestration.

Lives above ``cache`` and ``routes`` in the import graph so it can wire them
together without forcing either to depend on the other for its prewarm needs.
``app.py`` calls these from the ASGI lifespan; nothing else should import from
here at module scope.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from concurrent.futures import Future

import mteb
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

logger = logging.getLogger(__name__)

# Cap how many summaries build concurrently during preload. Each in-flight
# build holds a full pydantic schema + a worker thread for gzip; unbounded
# gather on ~50+ benchmarks spikes memory and the thread pool.
_PRELOAD_CONCURRENCY = 4


def _prewarm_training_datasets() -> None:
    """Populate ``_training_datasets_cached`` for every registered model.

    Without this, the first summary build pays ~2.5s on ``_collect_similar_tasks``
    per first-seen model.
    """
    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="warm-td") as ex:
        list(ex.map(_training_datasets_cached, MODEL_REGISTRY))


def _prewarm_list_schemas() -> None:
    """Pre-build the unfiltered list schemas + their serialised bytes.

    The four builders are independent — run them on a small thread pool.
    """
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
    except (OSError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
        # Warmup failure isn't fatal — the routes will rebuild on first request.
        # Narrow set lets programmer errors (TypeError, AttributeError) surface.
        logger.warning("light warmup failed: %s: %s", type(exc).__name__, exc)


def preload_summaries_in_background() -> None:
    """Pre-build every benchmark summary + per-language schema on a daemon thread."""
    if not get_settings().preload:
        return

    def _run() -> None:
        # Preload every registered benchmark — including off-menu ones — so
        # hidden-benchmark requests also hit the warmed cache and the model
        # detail page (which iterates all benchmarks) is cold-path-free.
        all_names = [b.name for b in mteb.get_benchmarks()]

        expected = (
            OSError,
            ValueError,
            KeyError,
            AttributeError,
            pl.exceptions.PolarsError,
        )

        async def _build_one(name: str) -> None:
            try:
                await get_summary_bytes(name)
            except expected as exc:
                logger.warning(
                    "warmup summary: %s failed (%s: %s)",
                    name,
                    type(exc).__name__,
                    exc,
                )
            try:
                await get_per_language_bytes(name)
            except expected as exc:
                logger.warning(
                    "warmup per-language: %s failed (%s: %s)",
                    name,
                    type(exc).__name__,
                    exc,
                )

        async def _build_all() -> None:
            sem = asyncio.Semaphore(_PRELOAD_CONCURRENCY)

            async def _bounded(name: str) -> None:
                async with sem:
                    await _build_one(name)

            await asyncio.gather(*(_bounded(n) for n in all_names))

        asyncio.run(_build_all())

    threading.Thread(target=_run, name="mteb-api-preload", daemon=True).start()
