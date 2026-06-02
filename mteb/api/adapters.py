"""Caching wrappers around the schema ``from_*`` constructors.

The conversion from mteb domain objects to API schemas lives on the schemas
themselves (each ``Schema.from_<thing>`` classmethod in :mod:`mteb.api.schemas`).
This module only owns the process-wide schema *cache* — every endpoint that
emits a task / benchmark / model goes through the helpers here so we pay the
pydantic construction cost once per object and serve dict-lookup-fast warm
hits afterwards.

Schemas can be safely cached because the underlying mteb metadata is static
after import (registries built at import time, no runtime mutation).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.api.schemas import (
    BenchmarkSchema,
    MenuEntrySchema,
    ModelMetaSchema,
    TaskMetaSchema,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.abstasks.abstask import AbsTask
    from mteb.benchmarks._leaderboard_menu import MenuEntry as MtebMenuEntry
    from mteb.benchmarks.benchmark import Benchmark
    from mteb.models.model_meta import ModelMeta


# Process-wide schema caches keyed by name. Memory is bounded by the number of
# registered tasks/benchmarks/models, all of which are fixed at import time.
_task_schema_cache: dict[str, TaskMetaSchema] = {}
_benchmark_schema_cache: dict[str, BenchmarkSchema] = {}
_model_schema_base_cache: dict[str, ModelMetaSchema] = {}


def task_to_meta_schema(task: AbsTask | type[AbsTask]) -> TaskMetaSchema:
    """Return the cached :class:`TaskMetaSchema` for ``task``.

    Accepts either an instantiated task or the task *class* — the schema only
    reads class-level ``metadata`` attributes, so callers that just need
    metadata (every API endpoint that emits tasks) can pass the class ref
    from ``_TASKS_REGISTRY`` to skip the ``cls().filter_languages()`` cost.
    Construction is delegated to :meth:`TaskMetaSchema.from_task_metadata`;
    we only memoise the result by ``task.metadata.name``.
    """
    md = task.metadata
    cached = _task_schema_cache.get(md.name)
    if cached is not None:
        return cached
    schema = TaskMetaSchema.from_task_metadata(md)
    _task_schema_cache[md.name] = schema
    return schema


def benchmark_to_schema(b: Benchmark) -> BenchmarkSchema:
    """Return the cached :class:`BenchmarkSchema` for ``b`` (memoised by name)."""
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

    All fields except ``zero_shot_pct`` are static per model, so we cache the
    ``zs=-1`` "base" instance per model name and ``model_copy`` it when a
    caller supplies a real percentage. ``model_copy`` runs inside pydantic-core
    (Rust) without revalidating fields — significantly faster than
    re-constructing the schema for every (model, benchmark) pair the summary
    endpoint emits.
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
    """Convert a top-level sequence of mteb menu entries into API schemas."""
    return [MenuEntrySchema.from_menu_entry(e) for e in entries]


def prewarm_schema_caches() -> tuple[int, int, int]:
    """Pre-build every task / benchmark / model schema into the module caches.

    Runs at API startup so the first hit to ``/tasks``, ``/benchmarks``, or
    ``/models`` doesn't pay the dedupe + pydantic construction cost. After
    this returns, every later ``task_to_meta_schema(t)`` /
    ``benchmark_to_schema(b)`` / ``model_meta_to_schema(meta, ...)`` call is a
    dict lookup (or, for ``model_meta_to_schema`` with a non-default
    ``zero_shot_pct``, a pydantic ``model_copy``).

    Tasks are warmed from the *class* registry (``_TASKS_REGISTRY``) rather
    than ``mteb.get_tasks()`` — the schema only reads class-level metadata
    so the per-task ``cls().filter_languages()`` instantiation
    ``mteb.get_tasks()`` does is pure overhead at startup (saves ~2.5s).
    """
    import mteb
    from mteb.get_tasks import _TASKS_REGISTRY
    from mteb.models.model_implementations import MODEL_REGISTRY

    for cls in _TASKS_REGISTRY.values():
        task_to_meta_schema(cls)
    for bench in mteb.get_benchmarks(display_on_leaderboard=True):
        benchmark_to_schema(bench)
    for meta in MODEL_REGISTRY.values():
        model_meta_to_schema(meta, zero_shot_pct=None)
    return (
        len(_task_schema_cache),
        len(_benchmark_schema_cache),
        len(_model_schema_base_cache),
    )
