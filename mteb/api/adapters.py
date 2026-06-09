"""Caching wrappers around the schema ``from_*`` constructors.

Each ``Schema.from_<thing>`` does the actual conversion; this module memoises
the result so endpoints pay the pydantic construction cost once per object.
Safe because mteb's registries are static after import.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from mteb.api.schemas import (
    BenchmarkSchema,
    MenuEntrySchema,
    ModelMetaSchema,
    TaskMetaSchema,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from concurrent.futures import Future

    from mteb.abstasks.abstask import AbsTask
    from mteb.benchmarks._leaderboard_menu import MenuEntry as MtebMenuEntry
    from mteb.benchmarks.benchmark import Benchmark
    from mteb.models.model_meta import ModelMeta


_task_schema_cache: dict[str, TaskMetaSchema] = {}
_benchmark_schema_cache: dict[str, BenchmarkSchema] = {}
_model_schema_base_cache: dict[str, ModelMetaSchema] = {}


def task_to_meta_schema(task: AbsTask | type[AbsTask]) -> TaskMetaSchema:
    """Return the cached :class:`TaskMetaSchema` for ``task`` (class or instance)."""
    md = task.metadata
    cached = _task_schema_cache.get(md.name)
    if cached is not None:
        return cached
    schema = TaskMetaSchema.from_task_metadata(md)
    _task_schema_cache[md.name] = schema
    return schema


def scoped_task_meta_schema(task: AbsTask) -> TaskMetaSchema:
    """Like :func:`task_to_meta_schema`, but respects ``task.languages``.

    When a benchmark registers a task with a language restriction the instance's
    ``task.languages`` reflects the restriction; the unscoped base schema uses
    the full metadata union.
    """
    from mteb.languages import language_label

    base = task_to_meta_schema(task)
    labels = sorted({language_label(c) for c in task.languages if c})
    return base.model_copy(update={"languages": labels})


def benchmark_to_schema(b: Benchmark) -> BenchmarkSchema:
    """Return the cached :class:`BenchmarkSchema` for ``b``."""
    cached = _benchmark_schema_cache.get(b.name)
    if cached is not None:
        return cached
    schema = BenchmarkSchema.from_benchmark(b)
    _benchmark_schema_cache[b.name] = schema
    return schema


def model_meta_to_schema(
    meta: ModelMeta,
    *,
    zero_shot_pct: int | None = None,
) -> ModelMetaSchema:
    """Return a :class:`ModelMetaSchema` for ``meta`` with ``zero_shot_pct`` applied.

    Caches the ``zs=-1`` base instance and ``model_copy``s it when a caller
    supplies a real percentage — pydantic-core skips revalidation on copies.
    """
    name = meta.name or ""
    cached = _model_schema_base_cache.get(name)
    if cached is None:
        cached = ModelMetaSchema.from_model_meta(meta, zero_shot_pct=None)
        _model_schema_base_cache[name] = cached
    if zero_shot_pct is None:
        return cached
    return cached.model_copy(update={"zero_shot_pct": int(zero_shot_pct)})


def menus_to_schemas(entries: Sequence[MtebMenuEntry]) -> list[MenuEntrySchema]:
    """Convert mteb menu entries into API schemas."""
    return [MenuEntrySchema.from_menu_entry(e) for e in entries]


def prewarm_schema_caches() -> None:
    """Fill every task / benchmark / model schema cache.

    Pydantic v2 construction releases the GIL inside pydantic-core, so threading
    the per-object builds is a real speedup over the sequential loop.
    """
    import mteb
    from mteb.get_tasks import _TASKS_REGISTRY
    from mteb.models.model_implementations import MODEL_REGISTRY

    tasks = list(_TASKS_REGISTRY.values())
    benches = list(mteb.get_benchmarks(display_on_leaderboard=True))
    models = list(MODEL_REGISTRY.values())

    # All three batches submitted together so workers stay saturated across
    # phases instead of draining at each boundary — tasks dominate (~1700
    # entries), so models + benches happily backfill the pool's tail.
    with ThreadPoolExecutor(max_workers=16, thread_name_prefix="warm-schema") as ex:
        futures: list[Future[Any]] = [
            *(ex.submit(task_to_meta_schema, t) for t in tasks),
            *(ex.submit(benchmark_to_schema, b) for b in benches),
            *(ex.submit(model_meta_to_schema, m) for m in models),
        ]
        for f in futures:
            f.result()
