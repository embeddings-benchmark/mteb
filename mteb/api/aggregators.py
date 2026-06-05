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

import polars as pl

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
    from mteb.api.cache import _load_per_benchmark_frames

    bench = mteb.get_benchmark(name)
    bench_schema = benchmark_to_schema(bench)
    # Use the scoped variant: when a benchmark registers a task with a
    # language restriction (e.g. MIRACL pinned to ``['eng']``), the per-task
    # `languages` field reflects that restriction instead of leaking the
    # task class's full unscoped language union into the benchmark's
    # tasksMeta. The frontend's filter sidebar populates from this list.
    tasks_meta: list[TaskMetaSchema] = [scoped_task_meta_schema(t) for t in bench.tasks]

    frames, _ = _load_per_benchmark_frames()
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

    # Flat per-task means in the summary (cheap, drives existing UI
    # consumers without waiting). The (task, subset, language) grid
    # lives in `/v1/benchmarks/{name}/per-task` and is loaded lazily
    # by the frontend only when needed (language filter).
    per_task_rows: dict[str, dict[str, float]] = {}
    if "No results" not in per_task_pl.columns and "Model" in per_task_pl.columns:
        task_cols_pt = [c for c in per_task_pl.columns if c != "Model"]
        for prow in per_task_pl.iter_rows(named=True):
            short = _parse_model_cell(prow["Model"])
            full = short_to_full.get(short, short)
            per_task_rows[full] = {
                col: float(v) for col in task_cols_pt if (v := prow[col]) is not None
            }

    # Per-(model, task) `trained_on` flag — written into the parquet by
    # `_build_pre_agg_df` so we don't have to reload ModelMeta here. Collapse
    # the long frame into {model_name -> sorted[task_name]} of tasks the model
    # was trained on. Empty list when the column isn't in the cache (older
    # parquet versions) or the model declares no overlap.
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

    # Identify which rank column to use. The mean-task-type builder
    # (MIEB, ViDoRe-style) writes BOTH "Rank" (the primary, assigned
    # in sort-by-Mean-(Task) order) and "Rank (Borda)" (kept for back-
    # compat). Prefer the explicit "Rank" — falling back to "Rank
    # (Borda)" first would shuffle MIEB rows so the actual top model
    # showed as e.g. #11. Standard builders only emit "Rank (Borda)",
    # so the fallback still applies there.
    summary_cols = summary_pl.columns
    rank_col = next(
        (c for c in ("Rank", "Rank (Mean Task)", "Rank (Borda)") if c in summary_cols),
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
                trained_on_tasks=trained_on_by_model.get(full, []),
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


async def build_benchmark_per_language(name: str) -> BenchmarkPerLanguageSchema:
    """Per-(model, language) mean main_score for one benchmark.

    Builds from the long results frame: explode the per-row ``language``
    list so each (task, subset) contributes one entry per language it
    covers, then group by (model_name, language) and mean. Codes are
    mapped to human labels via ``language_label`` so the keys line up
    with the column labels PerLanguageTab renders.
    """
    import mteb
    from mteb.api.cache import _load_per_benchmark_frames
    from mteb.languages import language_label

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

    # Explode the list-typed `language` column → one (model, task, subset, lang)
    # per row; then mean across all those score samples per (model, language).
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
        label = language_label(str(code)) if code != "Unknown" else "Unknown"
        rows.setdefault(mn, {})[label] = float(score)

    return BenchmarkPerLanguageSchema(
        benchmark_name=bench.name,
        rows=[
            BenchmarkPerLanguageRowSchema(model_name=mn, scores_by_language=s)
            for mn, s in rows.items()
        ],
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

    from mteb.api.cache import _load_per_benchmark_frames
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
    _, all_df = _load_per_benchmark_frames()
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


# ---------------------------------------------------------------------------
# /benchmarks/{name}/leaders — slim per-size-bucket top model.
# ---------------------------------------------------------------------------

# `(min, max)` ranges sized in MILLIONS of parameters (matches the
# `/benchmarks/{name}/leaders?buckets=…` wire format). `max=None`
# means "open-ended" (>= min). Bucket order in the request is preserved.
Bucket = tuple[float, float | None]

# `total_params_b` on `ModelMetaSchema` stays in billions for
# backwards compatibility with `/scores`. Convert each bucket's
# millions → billions exactly once before scanning rows.
_M_PER_B = 1000.0


def _row_score(row: SummaryRowSchema) -> float | None:
    """Pick whichever mean column the benchmark's builder populates.

    Standard benchmarks emit ``Mean (Task)`` (→ ``mean_task`` on the
    schema). MIEB / ViDoRe-style builders only populate
    ``mean_task_type`` because their primary aggregate is the mean
    of per-type means — `mean_task` is left null. Falling back keeps
    `_pick_leader` honest across both shapes.
    """
    if row.mean_task is not None:
        return row.mean_task
    return row.mean_task_type


def _pick_leader(
    rows: list[SummaryRowSchema], lo_b: float, hi_b: float | None
) -> tuple[SummaryRowSchema | None, float | None]:
    """Pick the highest-scoring row inside ``[lo_b, hi_b)`` (billions).

    Returns ``(row, score)`` or ``(None, None)`` if the bucket is
    empty. Excludes rows with `total_params_b <= 0` (unknown /
    missing). The upper bound is exclusive so callers can stitch
    buckets together without double-counting boundary models (e.g.
    a 1 B model belongs in `[0.5, 1)` for one query and `[1, 5)` in
    the next).
    """
    best: SummaryRowSchema | None = None
    best_score: float | None = None
    for r in rows:
        if r.total_params_b <= 0:
            continue
        if r.total_params_b < lo_b:
            continue
        if hi_b is not None and r.total_params_b >= hi_b:
            continue
        score = _row_score(r)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best = r
    return best, best_score


async def build_benchmark_leaders(
    name: str, buckets: list[Bucket]
) -> BenchmarkLeadersSchema:
    """Build the per-size-bucket leaders payload for a benchmark.

    Reuses the already-cached :class:`BenchmarkSummarySchema` (computed
    on demand by `get_summary`), so the additional cost vs. an
    on-demand full payload is just a per-bucket linear scan plus the
    slim schema construction — typically a few hundred bytes vs.
    megabytes for `/scores` on a multilingual benchmark.

    Buckets arrive in MILLIONS of parameters (matching the request
    wire format) and are converted to billions for the row scan;
    the response echoes back the original million-unit bounds so the
    caller doesn't have to remember the conversion.
    """
    # Local import to mirror the `build_model_scores` pattern above —
    # avoids a top-level cycle since `mteb.api.cache` imports
    # `build_benchmark_summary` from this module.
    from mteb.api.cache import get_summary

    summary = await get_summary(name)
    out_buckets: list[BucketLeaderSchema] = []
    for lo_m, hi_m in buckets:
        lo_b = lo_m / _M_PER_B
        hi_b = hi_m / _M_PER_B if hi_m is not None else None
        row, score = _pick_leader(summary.rows, lo_b, hi_b)
        leader: LeaderRowSchema | None = None
        if row is not None:
            leader = LeaderRowSchema(
                rank=row.rank,
                model=LeaderModelSchema(
                    name=row.model.name,
                    model_type=row.model.model_type,
                ),
                # `mean_task` here is the score that was used for the
                # bucket comparison — falls back to `mean_task_type`
                # when the benchmark builder didn't emit a Mean (Task).
                # Keeps the frontend rendering "Leader: … · {score}"
                # honest across both builder shapes.
                mean_task=score,
                total_params_b=row.total_params_b,
            )
        out_buckets.append(BucketLeaderSchema(min=lo_m, max=hi_m, leader=leader))
    return BenchmarkLeadersSchema(benchmark_name=name, buckets=out_buckets)
