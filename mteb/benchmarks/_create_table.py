from __future__ import annotations

import functools
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

from mteb.benchmarks.benchmark import _PRIMARY_METRIC_PRIORITY, BenchmarkAggregation
from mteb.get_tasks import _TASKS_REGISTRY
from mteb.models.model_implementations import MODEL_REGISTRY


@dataclass(frozen=True, slots=True)
class SummaryTable:
    """Output of every ``_create_summary_table_*`` builder.

    Carries the polars frame *plus* pointers to which columns hold the rank
    and the means. Consumers (API, leaderboard) read via these pointers
    instead of guessing from column names. Empty-result builds set
    ``is_empty=True`` and consumers short-circuit.

    The column-name strings here are the actual names present in ``df`` —
    no renames happen at consumption time. Defaults match the most common
    builder shape (the default per-task pivot branch of ``_create_summary_table``).

    Attributes:
        df: The summary polars frame.
        rank_col: Column in ``df`` holding the integer rank (1 = best).
        primary_metric_col: Column in ``df`` holding the primary scalar metric
            the rank is based on — varies per builder ("Mean (Task)",
            "Mean (Subset)", "Mean (TaskType)", or "Mean (Public)").
        task_type_mean_col: Column holding the mean-over-per-type-means, or
            ``None`` when the builder doesn't compute one.
        mean_public_col: Public split column, or ``None`` when the primary
            metric IS the public mean (no separate breakdown).
        mean_private_col: Private split column, or ``None``.
        is_empty: True for the ``No results`` sentinel — consumers short-circuit.
    """

    df: pl.DataFrame
    rank_col: str = "Rank (Borda)"
    primary_metric_col: str = "Mean (Task)"
    task_type_mean_col: str | None = "Mean (TaskType)"
    mean_public_col: str | None = None
    mean_private_col: str | None = None
    is_empty: bool = False


def _no_results_summary() -> SummaryTable:
    """Empty-result sentinel returned by every builder when the input is empty."""
    return SummaryTable(df=_no_results_frame(), is_empty=True)


@functools.lru_cache(maxsize=4096)
def _training_datasets_cached(model_name: str) -> frozenset[str] | None:
    """Memoized training datasets (with similar tasks) for a model.

    The similar-task graph traversal in ``ModelMeta.get_training_datasets()`` is
    expensive and depends only on the model, so cache it per model name here at the
    leaderboard layer (rather than polluting ``ModelMeta``). Both the summary's
    zero-shot column and ``_filter_models``' zero-shot check share this cache.

    Reads ``MODEL_REGISTRY`` directly (skips the rename check + KeyError path in
    ``get_model_meta``) — this is a hot-path lookup.
    """
    meta = MODEL_REGISTRY.get(model_name)
    if meta is None:
        return None
    training_datasets = meta.get_training_datasets()
    if training_datasets is None:
        return None
    return frozenset(training_datasets)


@functools.lru_cache(maxsize=4096)
def _zero_shot_pct_cached(model_name: str, task_names: tuple[str, ...]) -> int | None:
    """Memoized zero-shot percentage for a model over the given task names."""
    if not task_names:
        return None
    training_datasets = _training_datasets_cached(model_name)
    if training_datasets is None:
        return None
    overlap = training_datasets & set(task_names)
    return int(100 - 100 * (len(overlap) / len(task_names)))


def _is_zero_shot_cached(
    model_name: str, task_name_set: set[str] | frozenset[str]
) -> bool | None:
    """Cached equivalent of ``ModelMeta.is_zero_shot_on(task_names)`` for the leaderboard.

    Returns True if the model was not trained on any of the given tasks, False if it
    was, or None when the model has no training-data info. Reuses
    :func:`_training_datasets_cached`, so repeat calls (e.g. across model-filter
    interactions) avoid recomputing the similar-task graph traversal.
    """
    if not task_name_set:
        return True
    training_datasets = _training_datasets_cached(model_name)
    if training_datasets is None:
        return None
    return not bool(training_datasets & task_name_set)


@functools.cache
def _no_results_frame() -> pl.DataFrame:
    """The placeholder frame returned when an empty selection would have no rows.

    Cached because the frame is shape/content-immutable and consumers only
    inspect it; allocating a fresh frame per empty-input call wastes work
    during warmup of newly-added (still-empty) benchmarks.
    """
    return pl.DataFrame({"No results": ["You can try relaxing your criteria"]})


def _skipna_false_mean(cols: list[str]) -> pl.Expr:
    """Row-wise mean that returns null if any of `cols` is null.

    Matches `pd.DataFrame.mean(axis=1, skipna=False)` semantics. Uses polars'
    native single-pass `mean_horizontal(ignore_nulls=False)` instead of the
    older two-pass `any_horizontal(is_null) + when/otherwise + mean_horizontal`
    pattern — same result, half the horizontal scans.
    """
    return pl.mean_horizontal(cols, ignore_nulls=False)


def _mean_or_null(cols: list[str], alias: str) -> pl.Expr:
    """Skipna-false mean, with a Float64-null fallback for an empty column list.

    `_skipna_false_mean([])` (and `pl.mean_horizontal([], ...)`) raises a
    `ComputeError` because polars can't infer the output row count from no
    inputs. This helper substitutes a typed all-null column of the right
    dtype so the surrounding `select` can stay branch-free when a partition
    (e.g. public/private) happens to be empty for this benchmark.
    """
    if cols:
        return _skipna_false_mean(cols).alias(alias)
    return pl.lit(None).cast(pl.Float64).alias(alias)


def _get_borda_rank(score_cols: list[str]) -> pl.Expr:
    """Borda rank for each row across ``score_cols``, as a polars expression.

    Wide-form implementation: per-column rank (higher score → lower rank
    number) converted to a borda count (``n - rank``), summed row-wise, and
    ranked again with ``method="min"``. The row count ``n`` comes from the
    evaluation context via ``pl.len()``.

    Used by :meth:`Benchmark.to_dataframe`, which starts from a wide pandas
    frame and benefits from the inline-expression form. All internal summary
    builders use :func:`_borda_rank_from_long` instead, which replaces the
    N-element horizontal-rank expression tree with a single window function
    over the partition column.
    """
    n = pl.len()
    return (
        pl.sum_horizontal(
            [n - pl.col(c).rank(method="average", descending=True) for c in score_cols]
        )
        .rank(method="min", descending=True)
        .cast(pl.Int64)
    )


def _borda_rank_from_long(
    pl_df: pl.DataFrame,
    *,
    partition_col: str = "task_name",
    model_col: str = "model_name",
    score_col: str = "score",
    n_models: int | None = None,
) -> pl.DataFrame:
    """Long-form Borda rank — returns ``(model_name, "Rank (Borda)")``.

    Replaces the wide-form ``_get_borda_rank`` for callers with the long
    results frame. For a 100-task benchmark, the wide form builds a 100-node
    horizontal expression tree; this version uses a single ``rank().over()``
    window partitioned by ``task_name`` instead.

    Matches the wide-form semantics: each model earns
    ``n_models - rank_in_task`` points per task it ran (more wins = more
    points). Missing tasks contribute 0 (the model just has no row in that
    partition).

    ``n_models`` should equal the wide-form's ``pl.len()`` over the post-
    filter per_task / per_language frame — i.e. the count of models that
    survived the all-null-rows filter. Pass it explicitly when the long
    frame includes models that got filtered out of the wide pivot (e.g.
    per-language path where a model's entire language list explodes to
    null scores). When ``None``, defaults to ``pl_df``'s distinct model
    count, which matches when no models are all-null.
    """
    if n_models is None:
        n_models = pl_df.get_column(model_col).n_unique()
    return (
        pl_df.lazy()
        .group_by([model_col, partition_col])
        .agg(pl.col(score_col).mean())
        .with_columns(
            pl.col(score_col)
            .rank(method="average", descending=True)
            .over(partition_col)
            .alias("_r")
        )
        .group_by(model_col)
        .agg((n_models - pl.col("_r")).sum().alias("_b"))
        .with_columns(
            pl.col("_b")
            .rank(method="min", descending=True)
            .cast(pl.Int64)
            .alias("Rank (Borda)")
        )
        .select(model_col, "Rank (Borda)")
        .collect()
    )


def _split_on_capital(s: str) -> str:
    """Splits on capital letters and joins with spaces

    Returns:
        The input string split on capital letters and joined with spaces as a string.
    """
    return " ".join(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", s))


def _format_n_parameters(n_parameters: float | int | None) -> float | None:
    """Convert a parameter count to billions with 1M-precision (7M -> 0.007, 1.5B -> 1.5, None -> None)."""
    if n_parameters is None:
        return None
    return round(float(n_parameters) / 1e9, 3)


def _format_max_tokens(max_tokens: float | None) -> int | None:
    if max_tokens is None or max_tokens == np.inf:
        return None
    return int(max_tokens)


def _get_embedding_size(embed_dim: int | Sequence[int] | None) -> int | None:
    if embed_dim is None:
        return None
    if isinstance(embed_dim, int):
        return int(embed_dim)
    if isinstance(embed_dim, Sequence) and len(embed_dim) > 0:
        return int(max(embed_dim))
    return None


def _build_per_task_pivot(
    pl_df: pl.DataFrame,
) -> tuple[pl.DataFrame, list[str]] | None:
    """Pivot the long results frame to one row per model × one col per task.

    Returns ``(per_task, task_cols)`` or ``None`` for the three empty-input
    cases (empty frame, no ``model_name``, no tasks, or all-null rows). Every
    summary/per-task builder opens with this pattern — extracted so the four
    builders + per-task table builder don't duplicate the boilerplate.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return None
    per_task = (
        pl_df.group_by(["model_name", "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index="model_name", values="score")
    )
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return None
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return None
    return per_task, task_cols


def _get_means_per_types(
    task_cols: list[str],
) -> tuple[list[pl.Expr], list[str]]:
    """Per-task-type mean expressions for a given task-column set.

    Returns ``(type_exprs, type_cols)``: a list of polars expressions (one per task
    type, each aliased with the raw CamelCase type name) and the matching
    column-name list. The expressions can be splatted into a ``select`` /
    ``with_columns`` on the wide task frame so we don't materialise an
    intermediate ``mean_per_type`` frame and don't need a join to bring the type
    means back into the summary pipeline. Means use ``skipna=False`` semantics
    (matches the prior pandas implementation). Humanising the column name for
    display (``BitextMining`` → ``Bitext Mining``) is the leaderboard styler's
    job — keeping the canonical name here means the API can use it as a key
    directly without round-tripping through ``_split_on_capital``.
    """
    task_names_per_type: dict[str, list[str]] = defaultdict(list)
    for task_name in task_cols:
        # Read from the registered class to skip instantiation (get_task() runs filter_languages()).
        task_type = _TASKS_REGISTRY[task_name].metadata.type
        task_names_per_type[task_type].append(task_name)

    type_cols: list[str] = []
    type_exprs: list[pl.Expr] = []
    for task_type, tasks in task_names_per_type.items():
        type_cols.append(task_type)
        type_exprs.append(_skipna_false_mean(tasks).alias(task_type))
    return type_exprs, type_cols


@functools.lru_cache(maxsize=1)
def _static_model_meta() -> dict[str, dict[str, Any]]:
    """Cached per-model metadata dict keyed by ``model_name``.

    Built once from ``MODEL_REGISTRY`` (which is static after import) so that
    repeat leaderboard renders reuse the same dict objects instead of
    re-constructing one per model on every call to :func:`_attach_model_metadata`.
    Zero-shot is not stored here — it depends on the active task set and is
    layered on per call.
    """
    return {
        name: {
            "Max Tokens": _format_max_tokens(m.max_tokens),
            "Embedding Dimensions": _get_embedding_size(m.embed_dim),
            "Total Parameters (B)": _format_n_parameters(m.n_parameters),
            "Active Parameters (B)": _format_n_parameters(m.n_active_parameters),
            "Release Date": str(m.release_date) if m.release_date else None,
            "_model_link": m.reference,
        }
        for name, m in MODEL_REGISTRY.items()
    }


@functools.cache
def _meta_df_no_zs() -> pl.DataFrame:
    """Polars frame view of :func:`_static_model_meta` — one row per registered model.

    Cached for the process lifetime; ``MODEL_REGISTRY`` is static after import.
    Used by :func:`_attach_model_metadata` for an inner join against the long
    results frame's model column — replaces the prior ``map_batches`` + Python
    per-row resolver loop.
    """
    return pl.DataFrame(
        [{"model_name": name, **meta} for name, meta in _static_model_meta().items()]
    )


@functools.lru_cache(maxsize=128)
def _meta_df_with_zs(task_names_key: tuple[str, ...]) -> pl.DataFrame:
    """Per-task-set meta frame with ``Zero-shot`` pre-computed.

    Cache size 128 ≈ benchmark count; each frame ~300 rows × 8 cols ≈ 10 KB.
    """
    return _meta_df_no_zs().with_columns(
        pl.col("model_name")
        .map_elements(
            lambda n: (
                -1
                if (z := _zero_shot_pct_cached(n, task_names_key)) is None
                else int(z)
            ),
            return_dtype=pl.Int64,
        )
        .alias("Zero-shot"),
    )


_STANDARD_META_COLS: tuple[str, ...] = (
    "Zero-shot",
    "Active Parameters (B)",
    "Total Parameters (B)",
    "Embedding Dimensions",
    "Max Tokens",
)


def _order_summary_cols(
    joint_table: pl.DataFrame,
    *,
    rank_col: str,
    mean_cols: Sequence[str],
    type_cols: Sequence[str],
    extra_trailing: Sequence[str] = (),
) -> pl.DataFrame:
    """Reorder ``joint_table`` into the canonical summary column layout.

    Layout: ``rank_col | Model | meta cols | mean_cols | type_cols |
    extra_trailing | Release Date``. Columns not present in ``joint_table``
    are silently dropped — keeps the four ``_create_summary_table_*``
    builders from each re-spelling the ordering.
    """
    ordering = [
        rank_col,
        "Model",
        *_STANDARD_META_COLS,
        *mean_cols,
        *type_cols,
        *extra_trailing,
        "Release Date",
    ]
    return joint_table.select([c for c in ordering if c in joint_table.columns])


def _attach_model_metadata(
    joint_table: pl.DataFrame,
    task_names_key: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Filter to models with valid metadata and attach the standard summary columns.

    Inner-joins meta columns (``Max Tokens``, ``Embedding Dimensions``, ``Total/Active
    Parameters (B)``, ``Release Date``) onto ``joint_table`` (which must have a
    ``model_name`` column), renames it to ``Model`` keeping the canonical
    ``org/name`` identifier, and optionally adds a ``Zero-shot`` column when
    ``task_names_key`` is provided. Presentation concerns (markdown links,
    short-name shortening) are handled by the leaderboard styler.

    Implemented as a native polars inner join against a pre-built metadata
    frame — replaces the prior ``map_batches`` + per-row Python resolver,
    saving ~300 Python calls and ~300 dict allocations per build.
    """
    meta_df = (
        _meta_df_no_zs() if task_names_key is None else _meta_df_with_zs(task_names_key)
    )
    return (
        joint_table.join(meta_df, on="model_name", how="inner")
        .drop("_model_link")
        .rename({"model_name": "Model"})
    )


@dataclass(frozen=True, slots=True)
class _SummaryMetadata:
    """Column pointers and default sort decision for a SummaryTable.

    Built once by
    [_summary_metadata][mteb.benchmarks._create_table._summary_metadata] from
    the benchmark's `aggregations`; consumed by
    [_finalize_summary][mteb.benchmarks._create_table._finalize_summary] to
    populate the canonical column order and the
    [SummaryTable][mteb.benchmarks._create_table.SummaryTable] pointers
    without re-deriving the same decisions inline.

    Attributes:
        mean_cols: Mean column names to surface, in display order.
        primary_metric_col: Column the rank is built on top of.
        task_type_mean_col: `"Mean (TaskType)"` when surfaced, else `None`.
        mean_public_col: `"Mean (Public)"` when surfaced as the public split
            of `Mean (Task)` (Vidore case), else `None` — when the public
            mean IS the primary metric (RTEB before the issue-3902 collapse)
            the split breakdown vanishes.
        mean_private_col: `"Mean (Private)"` when surfaced, else `None`.
        default_sort: Column to sort by when the caller's `sort_by` is
            `None` — `"Mean (Subset)"` for subset-weighted benchmarks (HUME)
            and `"Rank (Borda)"` otherwise.
    """

    mean_cols: tuple[str, ...]
    primary_metric_col: str
    task_type_mean_col: str | None
    mean_public_col: str | None
    mean_private_col: str | None
    default_sort: str


def _summary_metadata(
    aggregations: Sequence[BenchmarkAggregation],
) -> _SummaryMetadata:
    """Decide which mean columns to surface for the given aggregations.

    Single source for what's a derivative of `aggregations` alone:
    display-order `mean_cols`, `primary_metric_col` (via
    [_PRIMARY_METRIC_PRIORITY][mteb.benchmarks.benchmark._PRIMARY_METRIC_PRIORITY]),
    the three pointer columns (`task_type_mean_col` / `mean_public_col` /
    `mean_private_col`), and the `default_sort` column. Pulling these out of
    [_create_summary_table][mteb.benchmarks._create_table._create_summary_table]
    makes the table-build path linear and keeps the "what columns mean what"
    logic next to the enum spec.
    """
    has = set(aggregations)
    public_private = BenchmarkAggregation.PUBLIC_PRIVATE in has
    mean_task = BenchmarkAggregation.MEAN_TASK in has

    # Names from the enum spec — single source of truth for column literals.
    (mean_task_type_col,) = BenchmarkAggregation.MEAN_TASK_TYPE.summary_columns
    public_col, private_col = BenchmarkAggregation.PUBLIC_PRIVATE.summary_columns
    (mean_subset_col,) = BenchmarkAggregation.MEAN_SUBSET.summary_columns

    return _SummaryMetadata(
        mean_cols=tuple(
            col
            for agg in BenchmarkAggregation
            if agg in has
            for col in agg.summary_columns
        ),
        primary_metric_col=next(
            (
                agg.summary_columns[0]
                for agg in _PRIMARY_METRIC_PRIORITY
                if agg in has and agg.summary_columns
            ),
            "Rank (Borda)",
        ),
        task_type_mean_col=(
            mean_task_type_col if BenchmarkAggregation.MEAN_TASK_TYPE in has else None
        ),
        # `Mean (Public)` is surfaced as a split breakdown only when
        # `Mean (Task)` is the primary — otherwise the public mean IS the
        # primary and the breakdown collapses.
        mean_public_col=public_col if (public_private and mean_task) else None,
        mean_private_col=private_col if public_private else None,
        default_sort=(
            mean_subset_col
            if BenchmarkAggregation.MEAN_SUBSET in has
            else "Rank (Borda)"
        ),
    )


def _finalize_summary(
    joint_table: pl.DataFrame,
    *,
    task_cols: list[str],
    type_cols: list[str],
    metadata: _SummaryMetadata,
    sort_by: str | Sequence[str] | None,
    rank_column_name: str | None,
) -> SummaryTable:
    """Attach model metadata, sort, rank, order columns, wrap in SummaryTable.

    Args:
        joint_table: Polars frame holding the per-model joint columns
            (model_name + mean / per-type cols + ``Rank (Borda)``).
        task_cols: Raw task column names — passed to the model-metadata join
            so the cached ``Zero-shot`` column is keyed against the right set.
        type_cols: Per-task-type mean column names to surface after the
            means. Empty list hides them entirely.
        metadata: Column pointers + default sort (see
            [_SummaryMetadata][mteb.benchmarks._create_table._SummaryMetadata]).
        sort_by: Sort column(s). ``None`` sorts by ``metadata.default_sort``
            (``"Rank (Borda)"`` ascending, anything else descending). A
            string or sequence sorts by those columns descending and adds a
            1-indexed rank column named ``rank_column_name`` (default
            ``"Rank"``); ``Rank (Borda)`` stays as a trailing column.
        rank_column_name: Name for the 1-indexed rank column added when
            ``sort_by`` is set. Falls back to ``"Rank"`` when ``None``.

    Returns:
        SummaryTable: Ready-to-style summary with metadata attached and
            columns laid out in the canonical order.
    """
    joint_table = _attach_model_metadata(
        joint_table, task_names_key=tuple(sorted(task_cols))
    )
    if sort_by is not None:
        sort_cols = [sort_by] if isinstance(sort_by, str) else list(sort_by)
        rank_col = rank_column_name or "Rank"
        joint_table = joint_table.sort(
            sort_cols, descending=True, nulls_last=True
        ).with_columns((pl.int_range(0, pl.len()) + 1).cast(pl.Int64).alias(rank_col))
        extra_trailing: tuple[str, ...] = ("Rank (Borda)",)
    else:
        default_sort = metadata.default_sort
        joint_table = joint_table.sort(
            default_sort,
            descending=default_sort != "Rank (Borda)",
            nulls_last=True,
        )
        rank_col = "Rank (Borda)"
        extra_trailing = ()

    return SummaryTable(
        df=_order_summary_cols(
            joint_table,
            rank_col=rank_col,
            mean_cols=metadata.mean_cols,
            type_cols=type_cols,
            extra_trailing=extra_trailing,
        ),
        rank_col=rank_col,
        primary_metric_col=metadata.primary_metric_col,
        task_type_mean_col=metadata.task_type_mean_col,
        mean_public_col=metadata.mean_public_col,
        mean_private_col=metadata.mean_private_col,
    )


def _create_summary_table(  # noqa: PLR0914
    pl_df: pl.DataFrame,
    *,
    aggregations: Sequence[BenchmarkAggregation] | None = None,
    sort_by: str | Sequence[str] | None = None,
    rank_column_name: str | None = None,
) -> SummaryTable:
    """Build the leaderboard summary table for a benchmark.

    Computes every candidate mean column up-front, then drops the ones not
    requested by ``aggregations``. Stays in polars throughout (aggregation,
    pivot, type-means, Borda, metadata join, sort) — the leaderboard converts
    to pandas only at the styling boundary.

    Candidate columns produced (kept or dropped based on `aggregations`):

    - `Mean (Task)` — sample-weighted mean across tasks
        ([MEAN_TASK][mteb.benchmarks.benchmark.BenchmarkAggregation.MEAN_TASK]).
    - `Mean (TaskType)` — mean of per-task-type means
        ([MEAN_TASK_TYPE][mteb.benchmarks.benchmark.BenchmarkAggregation.MEAN_TASK_TYPE]).
    - One column per task type (e.g. `Retrieval`, `Classification`)
        ([TASK_TYPES][mteb.benchmarks.benchmark.BenchmarkAggregation.TASK_TYPES]).
    - `Mean (Public)` / `Mean (Private)` — public/private split, computed
        when the long frame carries `is_public` and kept when
        [PUBLIC_PRIVATE][mteb.benchmarks.benchmark.BenchmarkAggregation.PUBLIC_PRIVATE]
        is requested.
    - `Mean (Subset)` — subset-weighted mean across `(task, subset)` pairs,
        only computed when
        [MEAN_SUBSET][mteb.benchmarks.benchmark.BenchmarkAggregation.MEAN_SUBSET]
        is in `aggregations` (needs the `subset` column).

    Args:
        pl_df: Long polars frame with at least `model_name`, `task_name`,
            and `score` columns. `is_public` (optional) enables the
            public/private split; `subset` (required for `MEAN_SUBSET`)
            enables subset-weighted aggregation.
        aggregations: Sequence of
            [BenchmarkAggregation][mteb.benchmarks.benchmark.BenchmarkAggregation]
            members (forwarded from
            [Benchmark.aggregations][mteb.benchmarks.benchmark.Benchmark.aggregations]).
            Defaults to `(MEAN_TASK, MEAN_TASK_TYPE, TASK_TYPES)`.
        sort_by: Column(s) to sort rows by. `None` sorts by the default
            (`Mean (Subset)` when `MEAN_SUBSET` is requested, else
            `Rank (Borda)`). A string or sequence of strings sorts by those
            columns descending and adds a 1-indexed rank column named
            `rank_column_name` — `Rank (Borda)` stays as a trailing column.
        rank_column_name: Name for the 1-indexed rank column added when
            `sort_by` is set. Falls back to `"Rank"` when `None`.

    Returns:
        SummaryTable: One row per model — meta cols, the mean columns
            requested by `aggregations`, and a rank column.
            `primary_metric_col` points at the column the rank is built on
            top of so consumers don't have to guess.
    """
    if aggregations is None:
        aggregations = (
            BenchmarkAggregation.MEAN_TASK,
            BenchmarkAggregation.MEAN_TASK_TYPE,
            BenchmarkAggregation.TASK_TYPES,
        )
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_summary()

    want_task_types = BenchmarkAggregation.TASK_TYPES in aggregations
    want_subset = BenchmarkAggregation.MEAN_SUBSET in aggregations

    # --- per-task wide frame (always carries is_public when present) ---
    has_is_public = "is_public" in pl_df.columns
    per_task_aggs = [pl.col("score").mean()]
    if has_is_public:
        per_task_aggs.append(pl.col("is_public").first())
    per_task_long = pl_df.group_by(["model_name", "task_name"]).agg(*per_task_aggs)
    per_task = per_task_long.pivot(on="task_name", index="model_name", values="score")
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return _no_results_summary()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_summary()

    # --- public/private task partitions (treat all-public when missing) ---
    is_public_by_task: dict[str, bool] = (
        dict(per_task_long.select("task_name", "is_public").unique().iter_rows())
        if has_is_public
        else {}
    )
    public_present = [c for c in task_cols if is_public_by_task.get(c, True)]
    private_present = [c for c in task_cols if not is_public_by_task.get(c, True)]

    # --- compute every candidate column up-front ---
    # Column names come from the enum spec so adding/renaming an aggregation
    # only has to touch ``_AGGREGATION_SUMMARY_COLUMNS``.
    (mean_task_col,) = BenchmarkAggregation.MEAN_TASK.summary_columns
    (mean_task_type_col,) = BenchmarkAggregation.MEAN_TASK_TYPE.summary_columns
    public_col, private_col = BenchmarkAggregation.PUBLIC_PRIVATE.summary_columns

    type_exprs, type_cols = _get_means_per_types(task_cols)
    joint_table = per_task.select(
        "model_name",
        *type_exprs,
        _skipna_false_mean(task_cols).alias(mean_task_col),
        _mean_or_null(public_present, public_col),
        _mean_or_null(private_present, private_col),
    ).with_columns(_skipna_false_mean(type_cols).alias(mean_task_type_col))

    # Mean (Subset) + subset-partition Borda only when the benchmark actually
    # wants subset weighting — costs an extra group_by we'd otherwise skip.
    if want_subset:
        (mean_subset_col,) = BenchmarkAggregation.MEAN_SUBSET.summary_columns
        per_subset_long = pl_df.group_by(["model_name", "task_name", "subset"]).agg(
            pl.col("score").mean()
        )
        subset_mean = per_subset_long.group_by("model_name").agg(
            pl.col("score").mean().alias(mean_subset_col)
        )
        joint_table = joint_table.join(subset_mean, on="model_name", how="left")
        keyed = per_subset_long.with_columns(
            (pl.col("task_name") + "::" + pl.col("subset")).alias("_ts")
        )
        borda_df = _borda_rank_from_long(
            keyed, partition_col="_ts", n_models=per_task.height
        )
    else:
        borda_df = _borda_rank_from_long(pl_df, n_models=per_task.height)

    joint_table = joint_table.join(borda_df, on="model_name", how="left")

    drop_cols: list[str] = []
    for agg in BenchmarkAggregation:
        if agg in aggregations:
            continue
        if agg is BenchmarkAggregation.TASK_TYPES:
            drop_cols.extend(type_cols)
        else:
            drop_cols.extend(agg.summary_columns)
    if drop_cols:
        joint_table = joint_table.drop(drop_cols, strict=False)

    return _finalize_summary(
        joint_table,
        task_cols=task_cols,
        type_cols=type_cols if want_task_types else [],
        metadata=_summary_metadata(aggregations),
        sort_by=sort_by,
        rank_column_name=rank_column_name,
    )


def _create_per_task_table_from_benchmark_results(
    pl_df: pl.DataFrame,
    *,
    pivot: tuple[pl.DataFrame, list[str]] | None = None,
) -> pl.DataFrame:
    """Create per-task table from a long polars pre-aggregation frame.

    All aggregation, ranking, and sorting runs in polars; the result is converted to
    pandas only at the return boundary (the leaderboard's Styler is pandas-based).

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.
        pivot: Pre-computed (per_task, task_cols) shared with
            [_create_summary_table][mteb.benchmarks._create_table._create_summary_table].

    Returns:
        DataFrame with per-task scores, ready for styling in the leaderboard.
    """
    if pivot is None:
        pivot = _build_per_task_pivot(pl_df)
    if pivot is None:
        return _no_results_frame()
    per_task, task_cols = pivot

    borda_df = _borda_rank_from_long(pl_df, n_models=per_task.height)
    return (
        per_task.join(borda_df, on="model_name", how="left")
        .sort("Rank (Borda)")
        .rename({"model_name": "Model"})
        .select(["Model", *task_cols])
    )


def _create_per_language_table_from_benchmark_results(
    pl_df: pl.DataFrame,
    language_view: list[str] | Literal["all"],
) -> pl.DataFrame:
    """Create per-language table from a long polars pre-aggregation frame.

    Returns a DataFrame with one row per model and one column per language.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``language`` (list[str]),
            and ``score`` columns.
        language_view: List of languages to include, or ``"all"`` for every language
            present in the results.

    Returns:
        DataFrame with per-language scores, ready for styling in the leaderboard.
    """
    if language_view != "all" and not isinstance(language_view, list):
        raise ValueError("language_view must be a list of languages or 'all'")

    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    # Lazy pipeline so polars can fuse explode + filter + group_by. Project only
    # the columns we need so the explode has narrower rows. When a language subset
    # is selected, push the predicate *before* the explode by keeping only rows
    # whose language list intersects the selection — this avoids materialising
    # exploded rows we'll discard.
    lazy = pl_df.lazy().select("model_name", "language", "score")
    if language_view != "all":
        lazy = lazy.filter(
            pl.col("language").list.eval(pl.element().is_in(language_view)).list.any()
        )
    lazy = lazy.explode("language").drop_nulls("language")
    if language_view != "all":
        lazy = lazy.filter(pl.col("language").is_in(language_view))
    # Streaming engine handles the explode → group_by chain on tens of millions of
    # post-explode rows ~3-4× faster than the default in-memory engine here.
    lang_df = (
        lazy.group_by(["model_name", "language"])
        .agg(pl.col("score").mean())
        .collect(engine="streaming")
    )
    if lang_df.is_empty():
        return _no_results_frame()

    per_language = lang_df.pivot(on="language", index="model_name", values="score")
    lang_cols = [c for c in per_language.columns if c != "model_name"]
    if not lang_cols:
        return _no_results_frame()
    per_language = per_language.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in lang_cols])
    )
    if per_language.is_empty():
        return _no_results_frame()

    if len(lang_cols) == 1:
        per_language = per_language.sort(lang_cols[0], descending=True, nulls_last=True)
    else:
        # Long-form Borda over the (model, language) aggregate: single
        # window-rank instead of an N-column horizontal expression tree
        # (matters for multilingual benchmarks with ~80 language columns).
        # ``n_models = per_language.height`` matches the wide-form's
        # ``pl.len()`` — lang_df may contain models whose aggregated scores
        # are all null and got dropped from per_language.
        borda_df = _borda_rank_from_long(
            lang_df,
            partition_col="language",
            score_col="score",
            n_models=per_language.height,
        )
        per_language = (
            per_language.join(borda_df, on="model_name", how="left")
            .sort("Rank (Borda)")
            .drop("Rank (Borda)")
        )

    return per_language.rename({"model_name": "Model"}).select(["Model", *lang_cols])
