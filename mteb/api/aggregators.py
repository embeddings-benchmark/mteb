from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING

import polars as pl

import mteb
from mteb.api.adapters import (
    benchmark_to_schema,
    model_meta_to_schema,
    scoped_task_meta_schema,
    task_to_meta_schema,
)
from mteb.api.schemas import (
    BenchmarkLeadersSchema,
    BenchmarkPerLanguageRowSchema,
    BenchmarkPerLanguageSchema,
    BenchmarkSummarySchema,
    BucketLeaderSchema,
    LeaderModelSchema,
    LeaderRowSchema,
    ModelScoreRowSchema,
    ModelScoresSchema,
    SummaryRowSchema,
    TaskScoreRowSchema,
    TaskScoresSchema,
)
from mteb.benchmarks._create_table import _format_max_tokens
from mteb.get_tasks import _TASKS_REGISTRY
from mteb.languages import language_label
from mteb.models.model_implementations import MODEL_REGISTRY

if TYPE_CHECKING:
    from mteb.api.schemas import (
        BenchmarkSchema,
        ModelMetaSchema,
        TaskMetaSchema,
    )
    from mteb.cache.result_cache import ResultCache

logger = logging.getLogger(__name__)

_SUMMARY_META_COLS = frozenset(
    {
        "Rank (Borda)",
        "Rank (Mean Task)",
        "Rank",
        "Model",
        "Zero-shot",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        "Mean (Task)",
        "Mean (TaskType)",
        "Mean (Subset)",
        "Mean (Public)",
        "Mean (Private)",
        "Release Date",
    }
)


def _empty_summary(
    bench_name: str,
    bench_schema: BenchmarkSchema,
    tasks_meta: list[TaskMetaSchema],
) -> BenchmarkSummarySchema:
    """Empty-row summary used by every early-return path in build_benchmark_summary."""
    return BenchmarkSummarySchema(
        benchmark_name=bench_name,
        task_types=bench_schema.task_types,
        tasks=bench_schema.tasks,
        tasks_meta=tasks_meta,
        rows=[],
        aggregations=bench_schema.aggregations,
        show_zero_shot=bench_schema.show_zero_shot,
    )


def _per_task_rows_and_cols(
    per_task_pl: pl.DataFrame,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Index the per-task table by ``Model`` and surface its task columns.

    Returns ``({model -> {task: score}}, task_cols)``. Both empty when
    ``per_task_pl`` is the ``_no_results_frame`` sentinel or has no ``Model``
    column — collapses the duplicated empty-frame guard the summary builder
    used to apply at two separate call sites.
    """
    if "No results" in per_task_pl.columns or "Model" not in per_task_pl.columns:
        return {}, []
    task_cols = [c for c in per_task_pl.columns if c != "Model"]
    rows = {
        prow["Model"]: {
            col: float(v) for col in task_cols if (v := prow[col]) is not None
        }
        for prow in per_task_pl.iter_rows(named=True)
    }
    return rows, task_cols


def _extract_trained_on_map(long_df: pl.DataFrame) -> dict[str, list[str]]:
    """Build a ``model_name -> [task_name, ...]`` map (lists already sorted).

    Reads the long frame's pre-baked ``trained_on`` column (set by
    ``_build_pre_agg_df``). Aggregating + sorting in polars means the result
    is one column-aligned read — no per-row Python ``setdefault`` loop.
    """
    if "trained_on" not in long_df.columns:
        return {}
    grouped = (
        long_df.lazy()
        .filter(pl.col("trained_on"))
        .group_by("model_name")
        .agg(pl.col("task_name").unique().sort())
        .collect()
    )
    return dict(zip(grouped["model_name"].to_list(), grouped["task_name"].to_list()))


@functools.lru_cache(maxsize=128)
def _trained_on_map_cached(bench_name: str) -> dict[str, tuple[str, ...]]:
    """``model_name -> (task_name, ...)`` from the **unfiltered** per-benchmark frame.

    ``trained_on`` is a model/task property — invariant under the language
    filter. Caching by benchmark name lets language-filtered cold builds
    skip the polars groupby (~10–40 ms on big multilingual frames). Extra
    models in the map are harmless: the per-row caller only does
    ``map.get(model_in_summary_pl, [])``.

    Tuples (not lists) so the cached value is treated as immutable.
    """
    from mteb.api.frames import _load_per_benchmark_frames

    frames, _ = _load_per_benchmark_frames()
    long_df = frames.get(bench_name)
    if long_df is None or "trained_on" not in long_df.columns:
        return {}
    grouped = (
        long_df.lazy()
        .filter(pl.col("trained_on"))
        .group_by("model_name")
        .agg(pl.col("task_name").unique().sort())
        .collect()
    )
    return {
        mn: tuple(tasks)
        for mn, tasks in zip(
            grouped["model_name"].to_list(), grouped["task_name"].to_list()
        )
    }


@functools.lru_cache(maxsize=128)
def _benchmark_language_codes(bench_name: str) -> tuple[str, ...]:
    """Unique non-null language codes for ``bench_name``'s long frame.

    Cached for the process lifetime — the underlying per-benchmark frame is
    also ``functools.cache``-d in :func:`_load_per_benchmark_frames`, so the
    code set is immutable across calls. Lifts the explode + unique scan out
    of the language-filter hot path (~30 ms per call on the 8M-row
    multilingual frame).
    """
    from mteb.api.frames import _load_per_benchmark_frames

    frames, _ = _load_per_benchmark_frames()
    long_df = frames.get(bench_name)
    if long_df is None or "language" not in long_df.columns:
        return ()
    return tuple(
        long_df.lazy()
        .select(pl.col("language").explode().unique().drop_nulls())
        .collect(engine="streaming")["language"]
        .to_list()
    )


def _filter_long_df_by_languages(
    long_df: pl.DataFrame,
    languages: tuple[str, ...],
    bench_name: str,
) -> pl.DataFrame | None:
    """Filter ``long_df`` to rows whose ``language`` list intersects ``languages``.

    Returns the filtered frame, or ``None`` if no language code in the data
    matched any pick. The frontend filter holds labels (``"English"``) but
    the frame holds codes (``"eng-Latn"``); a code matches if it OR
    ``language_label(code)`` is in the pick set.

    Returns the input unchanged when ``languages`` is empty or the frame has
    no ``language`` column. The polars filter still uses ``list.eval`` — on
    the 8M-row multilingual frame it beats ``set_intersection`` (~30 %
    faster) and ``any_horizontal(contains)`` (which blows up at large pick
    sets); see ``scripts/bench_lang_filter.py``.
    """
    if not languages or "language" not in long_df.columns:
        return long_df
    picked_set = set(languages)
    code_match = [
        c
        for c in _benchmark_language_codes(bench_name)
        if c in picked_set or language_label(c) in picked_set
    ]
    if not code_match:
        return None
    return long_df.filter(
        pl.col("language").list.eval(pl.element().is_in(code_match)).list.any()
    )


def _recompute_lenient_means(
    scores_by_task: dict[str, float],
    task_to_type: dict[str, str],
) -> tuple[dict[str, float], float | None, float | None]:
    """Recompute per-type means, mean(task), and mean(task_type) leniently.

    Means are taken over only the tasks a model actually ran. Used in the
    language-filtered path so partial-coverage models don't all collapse to
    null. Per-task-type bucketing avoids letting a single dense type
    dominate (e.g. Classification with 20 tasks shouldn't outweigh
    Retrieval with 3).
    """
    type_buckets: dict[str, list[float]] = {}
    for tname, score in scores_by_task.items():
        ttype = task_to_type.get(tname)
        if ttype is None:
            continue
        type_buckets.setdefault(ttype, []).append(float(score))

    scores_by_task_type = {
        ttype: sum(vals) / len(vals) for ttype, vals in type_buckets.items() if vals
    }
    task_vals = list(scores_by_task.values())
    mean_task = sum(task_vals) / len(task_vals) if task_vals else None
    type_vals = list(scores_by_task_type.values())
    mean_type = sum(type_vals) / len(type_vals) if type_vals else None
    return scores_by_task_type, mean_task, mean_type


async def build_benchmark_summary(  # noqa: PLR0914
    name: str,
    cache: ResultCache,
    languages: tuple[str, ...] = (),
) -> BenchmarkSummarySchema:
    """Build the summary payload for one benchmark.

    Delegates to the benchmark's own ``_create_summary_table`` and the per-task
    table builder, then matches each row back onto ``MODEL_REGISTRY`` for
    structured ``ModelMetaSchema`` + per-task / per-type score maps.

    When ``languages`` is non-empty the long frame is pre-filtered to subsets
    whose ``language`` list intersects the picks (matched by raw code OR by
    ``language_label``) before the summary builders run.
    """
    from mteb.api.frames import _load_per_benchmark_frames

    bench = mteb.get_benchmark(name)
    bench_schema = benchmark_to_schema(bench)
    # Scoped variant: when a benchmark pins a shared task to specific languages
    # (e.g. MIRACL → ['eng']), the per-task `languages` reflects that pin.
    tasks_meta: list[TaskMetaSchema] = [scoped_task_meta_schema(t) for t in bench.tasks]

    frames, _ = _load_per_benchmark_frames()
    long_df = frames.get(bench.name)
    if long_df is None:
        # Benchmark missing from the parquet cache (newly added). Costly first
        # call; the parquet rebuild normally covers everything.
        results = cache.load_results(tasks=bench, require_model_meta=True)
        long_df = results._to_results_df(bench.tasks)

    if long_df.is_empty() or "model_name" not in long_df.columns:
        return _empty_summary(bench.name, bench_schema, tasks_meta)

    filtered = _filter_long_df_by_languages(long_df, languages, bench.name)
    if filtered is None:
        return _empty_summary(bench.name, bench_schema, tasks_meta)
    long_df = filtered

    # Compute the (model × task) wide pivot once on a worker thread and pass
    # it to both builders — they would otherwise each recompute it on their
    # own thread, doubling polars CPU on the heaviest single op. Subclass
    # builders that need an is_public-aware pivot ignore the kwarg and pay
    # the cost themselves; per-task table always benefits.
    pivot = await asyncio.to_thread(bench._build_per_task_pivot, long_df)
    summary, per_task_pl = await asyncio.gather(
        asyncio.to_thread(bench._create_summary_table, long_df, pivot=pivot),
        asyncio.to_thread(bench._create_per_task_table, long_df, pivot=pivot),
    )
    if summary.is_empty:
        return _empty_summary(bench.name, bench_schema, tasks_meta)
    summary_pl = summary.df

    # ``Model`` now holds the canonical org/name — no parsing needed.
    per_task_rows, task_cols_out = _per_task_rows_and_cols(per_task_pl)

    trained_on_by_model = _trained_on_map_cached(bench.name)

    # The SummaryTable wrapper tells us exactly which columns hold what — no
    # introspection needed. Type columns are still detected by exclusion since
    # those vary per benchmark (raw CamelCase like "Retrieval", "STS").
    type_cols = [c for c in summary_pl.columns if c not in _SUMMARY_META_COLS]

    # Language-filtered: switch to lenient means so partial-coverage models
    # don't all collapse to null. Unfiltered stays strict so partial-coverage
    # peers can't outrank full-coverage ones.
    language_filtered = bool(languages)
    task_to_type: dict[str, str] = (
        {tm.name: tm.type for tm in tasks_meta} if language_filtered else {}
    )

    rows: list[SummaryRowSchema] = []
    for idx, row in enumerate(summary_pl.iter_rows(named=True)):
        full = row["Model"]
        meta = MODEL_REGISTRY.get(full)
        if meta is None:
            logger.debug("Skipping %s — no MODEL_REGISTRY entry", full)
            continue

        zs_raw = row.get("Zero-shot")
        zs = int(zs_raw) if zs_raw is not None else None
        model_schema = model_meta_to_schema(meta, zero_shot_pct=zs)

        rank_raw = row.get(summary.rank_col)
        rank = int(rank_raw) if rank_raw is not None else idx + 1
        mean_task = row.get(summary.primary_metric_col)
        mean_type = (
            row.get(summary.task_type_mean_col) if summary.task_type_mean_col else None
        )
        scores_by_task_type = {
            col: v for col in type_cols if (v := row[col]) is not None
        }
        scores_by_task = per_task_rows.get(full, {})

        if language_filtered and scores_by_task:
            scores_by_task_type, mean_task, mean_type = _recompute_lenient_means(
                scores_by_task, task_to_type
            )

        rows.append(
            SummaryRowSchema.model_construct(
                rank=rank,
                model=model_schema,
                zero_shot_pct=model_schema.zero_shot_pct,
                active_params_b=model_schema.active_params_b,
                total_params_b=model_schema.total_params_b,
                embedding_dim=model_schema.embedding_dim,
                max_tokens=_format_max_tokens(model_schema.max_tokens),
                mean_task=mean_task,
                mean_task_type=mean_type,
                mean_public=(
                    row.get(summary.mean_public_col)
                    if summary.mean_public_col
                    else None
                ),
                mean_private=(
                    row.get(summary.mean_private_col)
                    if summary.mean_private_col
                    else None
                ),
                scores_by_task_type=scores_by_task_type,
                scores_by_task=scores_by_task,
                trained_on_tasks=list(trained_on_by_model.get(full, ())),
            )
        )

    return BenchmarkSummarySchema(
        benchmark_name=bench.name,
        task_types=type_cols,
        tasks=task_cols_out,
        tasks_meta=tasks_meta,
        rows=rows,
        aggregations=bench_schema.aggregations,
        show_zero_shot=bench_schema.show_zero_shot,
    )


async def build_benchmark_per_language(name: str) -> BenchmarkPerLanguageSchema:
    """Per-(model, language) mean main_score for one benchmark."""
    from mteb.api.frames import _load_per_benchmark_frames

    bench = mteb.get_benchmark(name)
    frames, _ = _load_per_benchmark_frames()
    long_df = frames.get(bench.name)
    if (
        long_df is None
        or long_df.is_empty()
        or "model_name" not in long_df.columns
        or "language" not in long_df.columns
    ):
        return BenchmarkPerLanguageSchema(benchmark_name=bench.name, rows=[])

    # asyncio.to_thread(lambda: ...) — collect() is CPU-bound (group_by over an exploded multilingual frame). The wrapper keeps the event loop responsive
    # so concurrent API requests aren't blocked.
    grouped = await asyncio.to_thread(
        lambda: long_df.lazy()
        .explode("language")
        .group_by(["model_name", "language"])
        .agg(pl.col("score").mean().alias("score"))
        .collect(engine="streaming")
    )

    rows: dict[str, dict[str, float]] = {}
    for row in grouped.iter_rows(named=True):
        mn = row["model_name"]
        code = row.get("language")
        score = row.get("score")
        if not mn or code is None or score is None:
            continue
        rows.setdefault(mn, {})[language_label(code)] = score

    return BenchmarkPerLanguageSchema(
        benchmark_name=bench.name,
        rows=[
            BenchmarkPerLanguageRowSchema.model_construct(
                model_name=mn, scores_by_language=s
            )
            for mn, s in rows.items()
        ],
    )


@functools.cache
def _task_to_hosting_benchmarks() -> dict[str, list[str]]:
    """Reverse index: task name -> list of benchmarks that include it."""
    out: dict[str, list[str]] = {}
    for bench in mteb.get_benchmarks(display_on_leaderboard=True):
        for t in bench.tasks:
            out.setdefault(t.metadata.name, []).append(bench.name)
    return out


def build_task_scores(name: str) -> TaskScoresSchema:
    """Per-model scores for a single task across every benchmark hosting it.

    ``score`` is the mean of the model's per-subset scores when it has covered
    every subset; ``null`` otherwise so partial-coverage models can't outrank
    fully-evaluated peers.
    """
    from mteb.api.frames import _load_per_benchmark_frames

    cls = _TASKS_REGISTRY[name]
    task_meta = task_to_meta_schema(cls)

    hosting_benchmarks = list(_task_to_hosting_benchmarks().get(name, ()))

    # Pull from the all-results frame so multilingual tasks aren't trimmed to
    # one benchmark's locale selection.
    _, all_df = _load_per_benchmark_frames()
    task_frame = all_df.filter(pl.col("task_name") == name)

    seen: dict[str, dict[str, float]] = {}
    all_subsets: set[str] = set()
    if not task_frame.is_empty():
        per_subset = (
            task_frame.drop_nulls("score")
            .group_by(["model_name", "subset"])
            .agg(pl.col("score").max())
        )
        for pr in per_subset.iter_rows(named=True):
            subset = str(pr["subset"])
            seen.setdefault(pr["model_name"], {})[subset] = float(pr["score"])
            all_subsets.add(subset)

    rows: list[TaskScoreRowSchema] = []
    for model_name, subset_scores in seen.items():
        meta = MODEL_REGISTRY.get(model_name)
        if meta is None:
            continue
        score: float | None
        if all_subsets <= subset_scores.keys():
            score = sum(subset_scores.values()) / len(subset_scores)
        else:
            score = None

        trained_on: bool | None
        if meta.training_datasets:
            trained_on = name in meta.training_datasets
        else:
            trained_on = None
        rows.append(
            TaskScoreRowSchema.model_construct(
                rank=0,
                model=model_meta_to_schema(meta),
                score=score,
                subset_scores=subset_scores,
                benchmarks=list(hosting_benchmarks),
                trained_on=trained_on,
            )
        )

    rows.sort(
        key=lambda r: (
            r.score is None,
            -(r.score if r.score is not None else 0.0),
            r.model.name,
        )
    )
    for i, row in enumerate(rows, start=1):
        row.rank = i

    return TaskScoresSchema(
        task=task_meta,
        benchmarks=hosting_benchmarks,
        subsets=sorted(all_subsets),
        rows=rows,
    )


# Per-benchmark ``{model_name -> row}`` index
_summary_row_indices: dict[str, dict[str, SummaryRowSchema]] = {}


def _summary_row_index(
    bench_name: str, summary: BenchmarkSummarySchema
) -> dict[str, SummaryRowSchema]:
    """Cached ``{model_name -> row}`` for ``summary``.

    Replaces the per-call ``next(r for r in summary.rows if r.model.name ==
    name)`` linear scan — once the index is built, every subsequent
    ``build_model_scores`` call (for any model) does an O(1) lookup instead
    of N pydantic-attribute accesses.
    """
    cached = _summary_row_indices.get(bench_name)
    if cached is None:
        cached = {r.model.name: r for r in summary.rows}
        _summary_row_indices[bench_name] = cached
    return cached


async def build_model_scores(name: str) -> ModelScoresSchema:
    """Per-benchmark scores for a single model.

    Fans out the cold summary builds with ``asyncio.gather`` so wall time is
    max() instead of sum(). Per-summary row lookup goes through
    :func:`_summary_row_index` — the first ``build_model_scores`` call after
    a process restart pays an O(N) index build per benchmark; every call
    after (for any model name) is O(1).

    Iterates every registered benchmark — including off-menu ones
    (``display_on_leaderboard=False``) — so submissions to hidden benchmarks
    still surface on the model detail page.
    """
    from mteb.api.cache import get_summary

    all_benchmarks = mteb.get_benchmarks()
    results = await asyncio.gather(
        *(get_summary(b.name) for b in all_benchmarks),
        return_exceptions=True,
    )

    model_meta: ModelMetaSchema | None = None
    rows: list[ModelScoreRowSchema] = []

    for bench, summary in zip(all_benchmarks, results):
        # gather(return_exceptions=True) substitutes failed tasks with their exception instance.
        if isinstance(summary, BaseException):
            continue
        row = _summary_row_index(bench.name, summary).get(name)
        if row is None:
            continue
        if model_meta is None:
            model_meta = row.model
        rows.append(
            ModelScoreRowSchema.model_construct(
                benchmark_name=bench.name,
                benchmark_display_name=bench.display_name or bench.name,
                rank=row.rank,
                total_models=len(summary.rows),
                mean_task=row.mean_task,
                mean_task_type=row.mean_task_type,
                zero_shot_pct=row.zero_shot_pct,
                task_types=summary.task_types,
                scores_by_task_type=row.scores_by_task_type,
            )
        )

    if model_meta is None:
        meta = MODEL_REGISTRY.get(name)
        if meta is None:
            raise KeyError(name)
        model_meta = model_meta_to_schema(meta)

    rows.sort(key=lambda r: r.rank)
    return ModelScoresSchema(model=model_meta, rows=rows)


Bucket = tuple[float, float | None]


def _pick_leader(
    rows: list[SummaryRowSchema], lo_b: float, hi_b: float | None
) -> SummaryRowSchema | None:
    """Best-ranked row inside ``[lo_b, hi_b)`` (billions).

    Best = lowest ``row.rank`` (Borda). Excludes rows with ``total_params_b
    <= 0``. Upper bound exclusive so buckets stitch without double-counting.
    """
    best: SummaryRowSchema | None = None
    best_rank: int | None = None
    for r in rows:
        if r.total_params_b is None or r.total_params_b <= 0 or r.total_params_b < lo_b:
            continue
        if hi_b is not None and r.total_params_b >= hi_b:
            continue
        if best_rank is None or r.rank < best_rank:
            best_rank = r.rank
            best = r
    return best


async def build_benchmark_leaders(
    name: str, buckets: list[Bucket]
) -> BenchmarkLeadersSchema:
    """Per-size-bucket leaders payload — reuses the cached summary."""
    from mteb.api.cache import get_summary

    summary = await get_summary(name)
    out_buckets: list[BucketLeaderSchema] = []
    for lo_m, hi_m in buckets:
        lo_b = lo_m / 1000.0
        hi_b = hi_m / 1000.0 if hi_m is not None else None
        row = _pick_leader(summary.rows, lo_b, hi_b)
        leader: LeaderRowSchema | None = None
        if row is not None:
            score = row.mean_task if row.mean_task is not None else row.mean_task_type
            leader = LeaderRowSchema.model_construct(
                rank=row.rank,
                model=LeaderModelSchema.model_construct(
                    name=row.model.name,
                    model_type=row.model.model_type,
                ),
                mean_task=score,
                total_params_b=row.total_params_b,
            )
        out_buckets.append(
            BucketLeaderSchema.model_construct(min=lo_m, max=hi_m, leader=leader)
        )
    return BenchmarkLeadersSchema(benchmark_name=name, buckets=out_buckets)
