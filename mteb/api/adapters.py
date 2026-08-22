"""Caching wrappers around the schema ``from_*`` constructors.

Memoised because mteb's registries are static after import.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, TypeVar

import mteb
from mteb.api.schemas import (
    BenchmarkSchema,
    MenuEntrySchema,
    ModelMetaSchema,
    TaskMetaSchema,
)
from mteb.api.settings import get_settings
from mteb.get_tasks import _TASKS_REGISTRY
from mteb.languages import language_label
from mteb.models.model_implementations import MODEL_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from concurrent.futures import Future

    from mteb.abstasks.abstask import AbsTask
    from mteb.benchmarks._leaderboard_menu import MenuEntry
    from mteb.benchmarks.benchmark import Benchmark
    from mteb.models.model_meta import ModelMeta


_T = TypeVar("_T")

_task_schema_cache: dict[str, TaskMetaSchema] = {}
_benchmark_schema_cache: dict[str, BenchmarkSchema] = {}
_model_schema_base_cache: dict[str, ModelMetaSchema] = {}


def _cached(cache: dict[str, _T], key: str, builder: Callable[[], _T]) -> _T:
    cached = cache.get(key)
    if cached is None:
        cached = builder()
        cache[key] = cached
    return cached


def task_to_meta_schema(task: AbsTask | type[AbsTask]) -> TaskMetaSchema:
    """Return the cached `TaskMetaSchema` for ``task`` (class or instance)."""
    md = task.metadata
    return _cached(
        _task_schema_cache, md.name, lambda: TaskMetaSchema.from_task_metadata(md)
    )


def scoped_task_meta_schema(task: AbsTask) -> TaskMetaSchema:
    """Like `task_to_meta_schema`, but respects benchmark-pinned ``task.languages``."""
    base = task_to_meta_schema(task)
    labels = sorted({language_label(c) for c in task.languages if c})
    return base.model_copy(update={"languages": labels})


def benchmark_to_schema(b: Benchmark) -> BenchmarkSchema:
    """Return the cached `BenchmarkSchema` for ``b``."""
    return _cached(
        _benchmark_schema_cache, b.name, lambda: BenchmarkSchema.from_benchmark(b)
    )


def model_meta_to_schema(
    meta: ModelMeta,
    *,
    zero_shot_pct: int | None = None,
) -> ModelMetaSchema:
    """Cached `ModelMetaSchema` for ``meta``; ``zero_shot_pct`` applied via ``model_copy``."""
    name = meta.name or ""
    cached = _cached(
        _model_schema_base_cache,
        name,
        lambda: ModelMetaSchema.from_model_meta(meta, zero_shot_pct=None),
    )
    if zero_shot_pct is None:
        return cached
    return cached.model_copy(update={"zero_shot_pct": int(zero_shot_pct)})


def menus_to_schemas(entries: Sequence[MenuEntry]) -> list[MenuEntrySchema]:
    """Convert mteb menu entries into API schemas."""
    return [MenuEntrySchema.from_menu_entry(e) for e in entries]


def prewarm_schema_caches() -> None:
    """Fill every task/benchmark/model schema cache; threaded since pydantic-core releases the GIL."""
    tasks = list(_TASKS_REGISTRY.values())
    benches = list(mteb.get_benchmarks(display_on_leaderboard=True))
    models = list(MODEL_REGISTRY.values())

    max_workers = get_settings().prewarm_max_workers
    with ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="warm-schema",
    ) as ex:
        futures: list[Future[Any]] = [
            *(ex.submit(task_to_meta_schema, t) for t in tasks),
            *(ex.submit(benchmark_to_schema, b) for b in benches),
            *(ex.submit(model_meta_to_schema, m) for m in models),
        ]
        for f in futures:
            f.result()
