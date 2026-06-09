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
from mteb.api.cache import _load_per_benchmark_frames, get_summary
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

# Non-score metadata columns from `_create_summary_table` — stripped when
# building `scores_by_task_type`.
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
    )


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

    if languages and "language" in long_df.columns:
        # The frontend filter holds labels ("English"); the frame holds codes
        # ("eng-Latn"). Map both ways and keep codes whose code or label is picked.
        picked_set = set(languages)
        all_codes_series = (
            long_df.lazy()
            .select(pl.col("language").explode().unique())
            .collect(engine="streaming")["language"]
        )
        code_match = [
            c
            for c in all_codes_series.to_list()
            if c is not None and (c in picked_set or language_label(c) in picked_set)
        ]
        if not code_match:
            return _empty_summary(bench.name, bench_schema, tasks_meta)
        long_df = long_df.filter(
            pl.col("language").list.eval(pl.element().is_in(code_match)).list.any()
        )

    # Both builders are independent and release the GIL inside polars — run
    # them concurrently so wall time is max() instead of sum().
    summary_pl, per_task_pl = await asyncio.gather(
        asyncio.to_thread(bench._create_summary_table, long_df),
        asyncio.to_thread(bench._create_per_task_table, long_df),
    )
    if "No results" in summary_pl.columns:
        return _empty_summary(bench.name, bench_schema, tasks_meta)

    # Summary frame replaces ``model_name`` with markdown-linked ``Model``;
    # rebuild a short→full map to look up MODEL_REGISTRY.
    short_to_full: dict[str, str] = {}
    for full_name in long_df.get_column("model_name").unique().to_list():
        short_to_full.setdefault(full_name.split("/")[-1], full_name)

    def _parse_model_cell(cell: str) -> str:
        if cell.startswith("[") and "](" in cell:
            return cell[1 : cell.index("](")]
        return cell

    per_task_rows: dict[str, dict[str, float]] = {}
    if "No results" not in per_task_pl.columns and "Model" in per_task_pl.columns:
        task_cols_pt = [c for c in per_task_pl.columns if c != "Model"]
        for prow in per_task_pl.iter_rows(named=True):
            short = _parse_model_cell(prow["Model"])
            full = short_to_full.get(short, short)
            per_task_rows[full] = {
                col: float(v) for col in task_cols_pt if (v := prow[col]) is not None
            }

    # `trained_on` is pre-baked into the parquet by `_build_pre_agg_df`.
    trained_on_by_model: dict[str, list[str]] = {}
    if "trained_on" in long_df.columns:
        trained_pl = (
            long_df.lazy()
            .filter(pl.col("trained_on"))
            .select(["model_name", "task_name"])
            .unique()
            .collect()
        )
        for mn, tn in trained_pl.iter_rows():
            trained_on_by_model.setdefault(mn, []).append(tn)
        for lst in trained_on_by_model.values():
            lst.sort()

    # Prefer "Rank" when present — MIEB/ViDoRe emit both "Rank" and
    # "Rank (Borda)" but Borda-first would shuffle MIEB rows.
    summary_cols = summary_pl.columns
    rank_col = next(
        (c for c in ("Rank", "Rank (Mean Task)", "Rank (Borda)") if c in summary_cols),
        None,
    )

    type_cols = [c for c in summary_cols if c not in _SUMMARY_META_COLS]
    # _create_summary_table humanises CamelCase ("BitextMining" → "Bitext
    # Mining"); strip the inserted spaces so the canonical keys match the
    # raw type names tasksMeta carries.
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
    has_public = "Mean (Public)" in summary_cols and mean_task_col != "Mean (Public)"
    has_private = "Mean (Private)" in summary_cols

    # Language-filtered: switch to lenient means so partial-coverage models
    # don't all collapse to null. Unfiltered stays strict so partial-coverage
    # peers can't outrank full-coverage ones.
    lenient_means = bool(languages)
    task_to_type: dict[str, str] = {}
    if lenient_means:
        from mteb.benchmarks._create_table import _split_on_capital

        task_to_type = {
            tm.name: _split_on_capital(tm.type).replace(" ", "") for tm in tasks_meta
        }

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

        if lenient_means and scores_by_task:
            # Lenient: mean over tasks the model actually ran; per-type bucket
            # then re-mean so dense-Classification doesn't dominate.
            type_buckets: dict[str, list[float]] = {}
            for tname, score in scores_by_task.items():
                ttype = task_to_type.get(tname)
                if ttype is None:
                    continue
                type_buckets.setdefault(ttype, []).append(float(score))
            scores_by_task_type = {
                ttype: sum(vals) / len(vals)
                for ttype, vals in type_buckets.items()
                if vals
            }
            if scores_by_task:
                vals = list(scores_by_task.values())
                mean_task = sum(vals) / len(vals)
            if scores_by_task_type:
                vals = list(scores_by_task_type.values())
                mean_type = sum(vals) / len(vals)

        # ``model_construct`` skips pydantic validation — fields are produced
        # internally with already-correct types so the per-row validation cost
        # (run ~500× per benchmark) is wasted work.
        rows.append(
            SummaryRowSchema.model_construct(
                rank=int(rank_value) if rank_value is not None else (idx + 1),
                model=model_schema,
                zero_shot_pct=model_schema.zero_shot_pct,
                active_params_b=model_schema.active_params_b,
                total_params_b=model_schema.total_params_b,
                embedding_dim=model_schema.embedding_dim,
                max_tokens=_format_max_tokens(model_schema.max_tokens),
                mean_task=float(mean_task) if mean_task is not None else None,
                mean_task_type=float(mean_type) if mean_type is not None else None,
                mean_public=float(mean_public) if mean_public is not None else None,
                mean_private=float(mean_private) if mean_private is not None else None,
                scores_by_task_type=scores_by_task_type,
                scores_by_task=scores_by_task,
                trained_on_tasks=trained_on_by_model.get(full, []),
            )
        )

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


async def build_benchmark_per_language(name: str) -> BenchmarkPerLanguageSchema:
    """Per-(model, language) mean main_score for one benchmark."""
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

    grouped = (
        long_df.lazy()
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
        rows.setdefault(mn, {})[language_label(str(code))] = float(score)

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


def build_task_scores(name: str, cache: ResultCache) -> TaskScoresSchema:
    """Per-model scores for a single task across every benchmark hosting it.

    ``score`` is the mean of the model's per-subset scores when it has covered
    every subset; ``null`` otherwise so partial-coverage models can't outrank
    fully-evaluated peers.
    """
    cls = _TASKS_REGISTRY[name]
    task_meta = task_to_meta_schema(cls)

    hosting_benchmarks = list(_task_to_hosting_benchmarks().get(name, ()))

    # Pull from the all-results frame so multilingual tasks aren't trimmed to
    # one benchmark's locale selection.
    _, all_df = _load_per_benchmark_frames()
    task_frame = all_df.filter(pl.col("task_name") == name)

    seen: dict[str, dict[str, float]] = {}
    if not task_frame.is_empty():
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
        score: float | None
        if all_subsets <= subset_scores.keys():
            score = sum(subset_scores.values()) / len(subset_scores)
        else:
            score = None
        # `None` when the model didn't declare its training datasets — the
        # parquet's `trained_on` column flattens "undeclared" and
        # "declared-but-clean" both to False, so go to the source.
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
    # Scored rows by score desc, null-score rows alphabetically at the bottom.
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


async def build_model_scores(name: str) -> ModelScoresSchema:
    """Per-benchmark scores for a single model.

    Fans out the cold summary builds with ``asyncio.gather`` so wall time is
    max() instead of sum(). A linear scan per benchmark is cheap (~500 rows ×
    50 benchmarks ≈ microseconds) so we don't memoise the per-summary index.

    Iterates every registered benchmark — including off-menu ones
    (``display_on_leaderboard=False``) — so submissions to hidden benchmarks
    still surface on the model detail page.
    """
    all_benchmarks = mteb.get_benchmarks()
    results = await asyncio.gather(
        *(get_summary(b.name) for b in all_benchmarks),
        return_exceptions=True,
    )

    model_meta: ModelMetaSchema | None = None
    rows: list[ModelScoreRowSchema] = []

    for bench, summary in zip(all_benchmarks, results):
        if isinstance(summary, BaseException):
            continue
        row = next((r for r in summary.rows if r.model.name == name), None)
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
        # Fall back to MODEL_REGISTRY so the response is meaningful even when
        # the model has no benchmark results yet.
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
        if (
            r.total_params_b is None
            or r.total_params_b is None
            or r.total_params_b <= 0
            or r.total_params_b < lo_b
        ):
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
