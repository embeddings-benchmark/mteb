"""Build :class:`BenchmarkSummarySchema` from real mteb results.

Bypasses ``_create_summary_table_from_benchmark_results`` because that helper
collapses ``model_name`` into a markdown-linked ``Model`` column for the Gradio
Styler. We need to keep the raw ``model_name`` so we can look up each row in
``MODEL_REGISTRY`` and emit a structured ``ModelMetaSchema``. We *do* reuse the
arithmetic primitives in ``_create_table`` so per-task-type means, the Borda
rank, and the zero-shot percentage stay identical to what the Gradio leaderboard
shows.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import TYPE_CHECKING

from mteb.api.adapters import (
    benchmark_to_schema,
    model_meta_to_schema,
    task_to_meta_schema,
)
from mteb.api.schemas import (
    BenchmarkSummarySchema,
    ModelScoreRowSchema,
    ModelScoresSchema,
    SummaryRowSchema,
    TaskScoreRowSchema,
    TaskScoresSchema,
)
from mteb.models.model_implementations import MODEL_REGISTRY

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.api.schemas import ModelMetaSchema, TaskMetaSchema
    from mteb.benchmarks._leaderboard_menu import MenuEntry
    from mteb.benchmarks.benchmark import Benchmark
    from mteb.cache.result_cache import ResultCache

logger = logging.getLogger(__name__)


async def build_benchmark_summary(  # noqa: PLR0914
    name: str,
    cache: ResultCache,
) -> BenchmarkSummarySchema:
    """Build the full summary payload the frontend expects for one benchmark.

    Delegates to the benchmark's own ``_create_summary_table`` (so RTEB/MIEB/
    ViDoRe subclasses get their custom aggregation) and to the existing
    per-task table builder, then matches each row back onto MODEL_REGISTRY so
    we can return structured ModelMetaSchema + per-task / per-type score maps.
    The long polars frame comes from the same parquet-backed cache the Gradio
    leaderboard uses, so the numbers here match the leaderboard byte-for-byte
    and we don't pay ``cache.load_results(tasks=bench)`` (~4s) per benchmark.
    The ``cache`` argument is kept for API compatibility and as a fallback for
    benchmarks not in the parquet cache.
    """
    import mteb
    from mteb.api.cache import get_all_benchmark_frames

    bench = mteb.get_benchmark(name)
    bench_schema = benchmark_to_schema(bench)
    tasks_meta: list[TaskMetaSchema] = [task_to_meta_schema(t) for t in bench.tasks]

    frames = get_all_benchmark_frames()
    long_df = frames.get(bench.name)
    if long_df is None:
        # Fallback for benchmarks missing from the parquet cache (e.g. recently
        # added). Costly on first call; the parquet rebuild path in
        # ``get_all_benchmark_frames`` should normally cover everything.
        results = cache.load_results(tasks=bench, require_model_meta=True)
        long_df = results.to_results_df(bench.tasks)

    if long_df.is_empty() or "model_name" not in long_df.columns:
        return BenchmarkSummarySchema(
            benchmark_name=bench.name,
            task_types=bench_schema.task_types,
            tasks=bench_schema.tasks,
            tasks_meta=tasks_meta,
            rows=[],
            aggregations=bench_schema.aggregations,
        )

    # Both builders aggregate the same long frame independently, so build them
    # in parallel — polars releases the GIL during heavy aggregations. Wall
    # time drops from sum(summary, per_task) to max(...) (typically 30-40%).
    # ``asyncio.to_thread`` dispatches onto the default executor; combined
    # with ``asyncio.gather`` it replaces the previous explicit
    # ThreadPoolExecutor without a per-request thread-pool spin-up.
    summary_pl, per_task_pl = await asyncio.gather(
        asyncio.to_thread(bench._create_summary_table, long_df),
        asyncio.to_thread(bench._create_per_task_table, long_df),
    )
    if "No results" in summary_pl.columns:
        return BenchmarkSummarySchema(
            benchmark_name=bench.name,
            task_types=bench_schema.task_types,
            tasks=bench_schema.tasks,
            tasks_meta=tasks_meta,
            rows=[],
            aggregations=bench_schema.aggregations,
        )

    # The summary frame replaces ``model_name`` with a markdown-linked ``Model``
    # column. We need the raw name to look up MODEL_REGISTRY, so rebuild a
    # ``short -> full`` map from the long frame's unique model names. ``.unique``
    # is much cheaper than the previous ``group_by(...).agg(pl.len())`` round
    # trip we used purely for ordering — we don't depend on order anywhere.
    short_to_full: dict[str, str] = {}
    for full_name in long_df.get_column("model_name").unique().to_list():
        short_to_full.setdefault(full_name.split("/")[-1], full_name)

    def _parse_model_cell(cell: str) -> str:
        # cell is either "[short](url)" markdown or a bare short name.
        if cell.startswith("[") and "](" in cell:
            return cell[1 : cell.index("](")]
        return cell

    # Build a per-model dict of task scores from the per_task frame. Iterating
    # the polars frame via ``iter_rows(named=True)`` skips the
    # ``to_pandas()`` + ``iterrows()`` overhead (the previous hot loop).
    per_task_rows: dict[str, dict[str, float]] = {}
    if "No results" not in per_task_pl.columns and "Model" in per_task_pl.columns:
        task_cols_pt = [c for c in per_task_pl.columns if c != "Model"]
        for prow in per_task_pl.iter_rows(named=True):
            short = _parse_model_cell(prow["Model"])
            full = short_to_full.get(short, short)
            per_task_rows[full] = {
                col: float(v) for col in task_cols_pt if (v := prow[col]) is not None
            }

    # Identify which rank column to use (the summary frames produced by the
    # various builders use one of these names — same logic as the Gradio styler).
    summary_cols = summary_pl.columns
    rank_col = next(
        (c for c in ("Rank (Borda)", "Rank (Mean Task)", "Rank") if c in summary_cols),
        None,
    )

    # Non-score metadata columns we strip out of `scores_by_task_type`.
    meta_cols = {
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
    type_cols = [c for c in summary_cols if c not in meta_cols]
    # mteb's _create_summary_table humanises type names via _split_on_capital
    # ("BitextMining" → "Bitext Mining"). tasksMeta carries the raw CamelCase
    # form, so the frontend's applyFilters silently drops every multi-word
    # type when comparing summary.taskTypes against tasksMeta[].type. Strip
    # the inserted spaces here so both sides agree on a single canonical key.
    type_cols_canonical = [c.replace(" ", "") for c in type_cols]
    canonical_to_display = dict(zip(type_cols_canonical, type_cols))
    mean_task_col = next(
        (
            c
            for c in ("Mean (Task)", "Mean (Subset)", "Mean (Public)")
            if c in summary_cols
        ),
        None,
    )
    mean_type_col = "Mean (TaskType)" if "Mean (TaskType)" in summary_cols else None
    # Split-aware benchmarks (ViDoRe family currently — RTEB renames these
    # away in its own wrapper) expose Mean (Public) / Mean (Private) as first-
    # class summary columns. Surface them when present so the frontend can
    # render the extra columns.
    has_public = "Mean (Public)" in summary_cols and mean_task_col != "Mean (Public)"
    has_private = "Mean (Private)" in summary_cols

    rows: list[SummaryRowSchema] = []
    for idx, row in enumerate(summary_pl.iter_rows(named=True)):
        short = _parse_model_cell(row["Model"])
        full = short_to_full.get(short, short)
        meta = MODEL_REGISTRY.get(full)
        if meta is None:
            logger.debug("Skipping %s — no MODEL_REGISTRY entry", full)
            continue

        zs_raw = row.get("Zero-shot")
        zs = int(zs_raw) if zs_raw is not None else None
        model_schema = model_meta_to_schema(meta, zero_shot_pct=zs)

        rank_value = row[rank_col] if rank_col else (idx + 1)
        mean_task = row[mean_task_col] if mean_task_col else None
        mean_type = row[mean_type_col] if mean_type_col else mean_task
        mean_public = row["Mean (Public)"] if has_public else None
        mean_private = row["Mean (Private)"] if has_private else None

        scores_by_task_type: dict[str, float] = {
            canonical: float(v)
            for canonical, display in canonical_to_display.items()
            if (v := row[display]) is not None
        }
        scores_by_task = per_task_rows.get(full, {})

        # mteb's _skipna_false_mean returns None when the row is missing a
        # single task or task-type cell. Pass that through as null — averaging
        # over partial coverage would surface a misleading number that ranks
        # alongside (or above) fully-evaluated peers. Clients render '—' for
        # null and sort null rows to the bottom.

        rows.append(
            SummaryRowSchema(
                rank=int(rank_value) if rank_value is not None else (idx + 1),
                model=model_schema,
                zero_shot_pct=model_schema.zero_shot_pct,
                active_params_b=model_schema.active_params_b,
                total_params_b=model_schema.total_params_b,
                embedding_dim=model_schema.embedding_dim,
                max_tokens=model_schema.max_tokens,
                mean_task=float(mean_task) if mean_task is not None else None,
                mean_task_type=float(mean_type) if mean_type is not None else None,
                mean_public=float(mean_public) if mean_public is not None else None,
                mean_private=float(mean_private) if mean_private is not None else None,
                scores_by_task_type=scores_by_task_type,
                scores_by_task=scores_by_task,
            )
        )

    # `task_cols` is whatever the per-task frame produced (real task names).
    task_cols_out: list[str] = []
    if "No results" not in per_task_pl.columns:
        task_cols_out = [c for c in per_task_pl.columns if c != "Model"]

    return BenchmarkSummarySchema(
        benchmark_name=bench.name,
        task_types=type_cols_canonical,
        tasks=task_cols_out,
        tasks_meta=tasks_meta,
        rows=rows,
        aggregations=bench_schema.aggregations,
    )


@functools.lru_cache(maxsize=1)
def _task_to_hosting_benchmarks() -> dict[str, list[str]]:
    """Reverse index: task name -> list of benchmark names that include it.

    Building this lazily on first call costs the same as a single
    ``build_task_scores`` invocation (and is amortised across all subsequent
    calls). Before this, every call iterated every benchmark and did an
    ``any()`` scan of its tasks — O(benchmarks × tasks_per_benchmark).
    """
    import mteb

    out: dict[str, list[str]] = {}
    for bench in mteb.get_benchmarks(display_on_leaderboard=True):
        for t in bench.tasks:
            out.setdefault(t.metadata.name, []).append(bench.name)
    return out


def build_task_scores(name: str, cache: ResultCache) -> TaskScoresSchema:
    """Per-model scores for a single task across all benchmarks that include it.

    For each model we emit:

    * ``score`` — mean of that model's per-subset scores (matches what the
      leaderboard's per-task column shows; tolerates partial subset coverage).
    * ``subset_scores`` — per-``hf_subset`` main score (real, not synthesised).

    Models with zero usable subset scores are dropped from the response so
    they can't sort to the top with a NaN/null score.

    Implementation: reuse the parquet-backed per-benchmark polars frames the
    Gradio leaderboard already loads (``get_all_benchmark_frames``) instead of
    re-scanning every JSON file under the result cache. The frame has one row
    per ``(model, task, split, subset, language)`` so taking
    ``max(score)`` per ``(model_name, subset)`` matches the previous
    "max across splits per ``hf_subset``" semantics from
    ``_extract_subset_scores``.
    """
    import polars as pl

    from mteb.api.cache import get_all_results_df
    from mteb.get_tasks import _TASKS_REGISTRY

    # Read directly from the registry — schema + metadata are class-level, so
    # the ``cls().filter_languages()`` work ``mteb.get_task`` does is pure
    # overhead here.
    cls = _TASKS_REGISTRY[name]
    task_meta = task_to_meta_schema(cls)

    # Reverse-indexed lookup (cached across all build_task_scores calls).
    hosting_benchmarks = list(_task_to_hosting_benchmarks().get(name, ()))

    # Pull straight from the all-results frame (every model × every subset
    # this task was ever scored on), no benchmark scoping. The previous
    # per-benchmark frames were each trimmed to their host benchmark's locale
    # selection, so multilingual tasks like AmazonReviewsClassification could
    # surface as little as one subset even when the underlying results
    # covered all six.
    all_df = get_all_results_df()
    task_frame = all_df.filter(pl.col("task_name") == name)

    seen: dict[str, dict[str, float]] = {}
    if not task_frame.is_empty():
        # Max across splits/languages per (model, hf_subset). Drop null
        # scores so a stray null can't outrank a real one through ``max``.
        per_subset = (
            task_frame.drop_nulls("score")
            .group_by(["model_name", "subset"])
            .agg(pl.col("score").max())
        )
        for pr in per_subset.iter_rows(named=True):
            seen.setdefault(pr["model_name"], {})[str(pr["subset"])] = float(
                pr["score"]
            )

    all_subsets: set[str] = set()
    for subset_scores in seen.values():
        all_subsets.update(subset_scores.keys())

    rows: list[TaskScoreRowSchema] = []
    for model_name, subset_scores in seen.items():
        meta = MODEL_REGISTRY.get(model_name)
        if meta is None:
            continue
        # Headline score is the mean of *every* subset this task offers — if
        # the model is missing any of them, leave the score null so it can't
        # outrank a fully-evaluated peer with a higher-but-narrower number.
        score: float | None
        if all_subsets <= subset_scores.keys():
            score = sum(subset_scores.values()) / len(subset_scores)
        else:
            score = None
        rows.append(
            TaskScoreRowSchema(
                rank=0,
                model=model_meta_to_schema(meta, zero_shot_pct=None),
                score=score,
                subset_scores=subset_scores,
                benchmarks=list(hosting_benchmarks),
            )
        )
    # Sort: scored rows by score desc, then null-score rows alphabetically
    # at the bottom. Rank is then assigned in display order.
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


@functools.lru_cache(maxsize=128)
def _flat_leaderboard_benchmarks() -> tuple[Benchmark, ...]:
    """Flattened, ordered tuple of every leaderboard ``Benchmark`` (cached)."""
    from mteb.benchmarks._leaderboard_menu import (
        GP_BENCHMARK_ENTRIES,
        R_BENCHMARK_ENTRIES,
    )
    from mteb.benchmarks.benchmark import Benchmark

    def _flatten(entries: Sequence[Benchmark | MenuEntry]) -> list[Benchmark]:
        out: list[Benchmark] = []
        for e in entries:
            if isinstance(e, Benchmark):
                out.append(e)
            else:
                out.extend(_flatten(e.benchmarks))
        return out

    return tuple(_flatten([*GP_BENCHMARK_ENTRIES, *R_BENCHMARK_ENTRIES]))


def _summary_row_index(
    summary: BenchmarkSummarySchema,
) -> dict[str, SummaryRowSchema]:
    """O(1) ``model_name -> SummaryRowSchema`` lookup for one cached summary."""
    return {row.model.name: row for row in summary.rows}


async def build_model_scores(name: str) -> ModelScoresSchema:
    """Per-benchmark scores for a single model.

    Each ``/models/{name}/scores`` request would otherwise sequentially trigger
    a cold summary build per benchmark (~4s × ~50 benchmarks ≈ 3 minutes on the
    first call). ``asyncio.gather`` fans out every benchmark summary onto the
    default thread executor (polars releases the GIL inside
    ``_create_summary_table``) so they overlap, and the per-benchmark
    ``model_name -> row`` index lets us pluck the target model in O(1)
    instead of scanning ``summary.rows`` (~700 entries) for each.
    """
    from mteb.api.cache import get_summary

    all_benchmarks = _flat_leaderboard_benchmarks()
    results = await asyncio.gather(
        *(get_summary(b.name) for b in all_benchmarks),
        return_exceptions=True,
    )

    model_meta: ModelMetaSchema | None = None
    rows: list[ModelScoreRowSchema] = []

    for bench, summary in zip(all_benchmarks, results):
        if isinstance(summary, BaseException):
            continue
        index = _summary_row_index(summary)
        row = index.get(name)
        if row is None:
            continue
        if model_meta is None:
            model_meta = row.model
        rows.append(
            ModelScoreRowSchema(
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
        # Fall back to MODEL_REGISTRY so /models/{name}/scores returns a
        # meaningful 200 even when the model has no benchmark results yet.
        meta = MODEL_REGISTRY.get(name)
        if meta is None:
            raise KeyError(name)
        model_meta = model_meta_to_schema(meta, zero_shot_pct=None)

    rows.sort(key=lambda r: r.rank)
    return ModelScoresSchema(model=model_meta, rows=rows)
