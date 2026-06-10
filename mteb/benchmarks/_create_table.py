from __future__ import annotations

import functools
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import polars as pl

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
    builder shape (the default ``_create_summary_table_from_benchmark_results``).

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
    """Row-wise mean that returns null if any of ``cols`` is null.

    Matches ``pd.DataFrame.mean(axis=1, skipna=False)`` semantics. Uses polars'
    native single-pass ``mean_horizontal(ignore_nulls=False)`` instead of the
    older two-pass ``any_horizontal(is_null) + when/otherwise + mean_horizontal``
    pattern — same result, half the horizontal scans.
    """
    return pl.mean_horizontal(cols, ignore_nulls=False)


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


def _build_joint_with_type_means_and_borda(
    per_task: pl.DataFrame,
    task_cols: list[str],
    pl_df: pl.DataFrame,
    *,
    task_type_mean_alias: str = "Mean (TaskType)",
    extra_select_exprs: Sequence[pl.Expr] = (),
) -> tuple[pl.DataFrame, list[str]]:
    """Compute per-type means + the task-type mean + Borda rank into one frame.

    Returns ``(joint_table, type_cols)`` where ``joint_table`` has
    ``model_name`` + per-type mean columns + ``extra_select_exprs`` +
    ``task_type_mean_alias`` (the mean of the per-type means) +
    ``Rank (Borda)``.

    Lazy chain so polars fuses select + with_columns into one query plan (no
    intermediate materialisation between the per-type means and the task-type
    mean that reads them). The Borda is computed long-form so
    ``n_models = per_task.height`` matches the wide-form's ``pl.len()`` —
    ``pl_df`` may contain all-null models that were filtered out of
    ``per_task``.
    """
    type_exprs, type_cols = _get_means_per_types(task_cols)
    borda_df = _borda_rank_from_long(pl_df, n_models=per_task.height)
    joint_table = (
        per_task.lazy()
        .select("model_name", *type_exprs, *extra_select_exprs)
        .with_columns(_skipna_false_mean(type_cols).alias(task_type_mean_alias))
        .collect()
        .join(borda_df, on="model_name", how="left")
    )
    return joint_table, type_cols


def _create_summary_table_from_benchmark_results(
    pl_df: pl.DataFrame,
    *,
    pivot: tuple[pl.DataFrame, list[str]] | None = None,
) -> SummaryTable:
    """Create summary table from a long polars pre-aggregation frame.

    Stays in polars throughout (aggregation, pivot, type-means, borda, model metadata,
    sort); the leaderboard converts to pandas at the styling boundary.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.
        pivot: Pre-computed (per_task, task_cols) from :func:`_build_per_task_pivot`.
            When the API builds summary + per-task tables together it computes
            this once and passes it to both, halving polars CPU on the pivot.

    Returns:
        SummaryTable wrapping a DataFrame with one row per model — meta cols,
        ``Mean (Task)`` (primary), ``Mean (TaskType)``, per-type cols.
    """
    if pivot is None:
        pivot = _build_per_task_pivot(pl_df)
    if pivot is None:
        return _no_results_summary()
    per_task, task_cols = pivot

    joint_table, type_cols = _build_joint_with_type_means_and_borda(
        per_task,
        task_cols,
        pl_df,
        extra_select_exprs=(_skipna_false_mean(task_cols).alias("Mean (Task)"),),
    )

    # Attach metadata BEFORE sorting: the attach step inner-joins on the model
    # registry and drops unknown models — sorting those rows would be wasted.
    joint_table = _attach_model_metadata(
        joint_table, task_names_key=tuple(sorted(task_cols))
    ).sort("Rank (Borda)")

    return SummaryTable(
        df=_order_summary_cols(
            joint_table,
            rank_col="Rank (Borda)",
            mean_cols=("Mean (Task)", "Mean (TaskType)"),
            type_cols=type_cols,
        ),
        rank_col="Rank (Borda)",
        primary_metric_col="Mean (Task)",
        task_type_mean_col="Mean (TaskType)",
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
            :func:`_create_summary_table_from_benchmark_results`.

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


@dataclass(frozen=True, slots=True)
class _PublicPrivateBuild:
    """Shared intermediate build for public/private-aware summary tables.

    Captures everything the two consumers
    (:func:`_create_summary_table_mean_public_private` and
    :class:`~mteb.benchmarks.benchmark.VidoreBenchmark._create_summary_table`)
    need: the joint table with per-type + public/private mean columns, the
    filtered wide pivot (for ``n_models`` on Borda), the long aggregate (for
    Borda ranking), the per-type mean column names, the sorted task-name key
    for metadata attachment, and the public/private task names actually
    present in the pivot (used by both the per-mean expressions above and any
    consumer that needs to filter Borda by visibility).
    """

    joint_table: pl.DataFrame
    per_task: pl.DataFrame
    task_cols: list[str]
    type_cols: list[str]
    per_task_long: pl.DataFrame
    public_present: list[str]
    private_present: list[str]


def _build_public_private_joint(pl_df: pl.DataFrame) -> _PublicPrivateBuild | None:
    """Build the public/private joint frame, or ``None`` for empty input.

    Steps shared by both public-private builders: group long results by
    (model, task) keeping ``is_public``; partition tasks into public/private;
    pivot to wide; drop all-null rows; compute per-type means + public/private
    means into one frame.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return None

    per_task_long = pl_df.group_by(["model_name", "task_name"]).agg(
        pl.col("score").mean(),
        pl.col("is_public").first(),
    )
    # One scan over the per-task aggregate instead of two: collect distinct
    # (task_name, is_public) pairs, then partition in Python.
    task_pub_pairs = per_task_long.select("task_name", "is_public").unique().iter_rows()
    public_tasks: list[str] = []
    private_tasks: list[str] = []
    for tname, is_pub in task_pub_pairs:
        (public_tasks if is_pub else private_tasks).append(tname)

    per_task = per_task_long.pivot(on="task_name", index="model_name", values="score")
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return None
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return None

    type_exprs, type_cols = _get_means_per_types(task_cols)

    public_present = [c for c in public_tasks if c in task_cols]
    private_present = [c for c in private_tasks if c in task_cols]

    public_mean_expr = (
        _skipna_false_mean(public_present).alias("Mean (Public)")
        if public_present
        else pl.lit(None).cast(pl.Float64).alias("Mean (Public)")
    )
    private_mean_expr = (
        _skipna_false_mean(private_present).alias("Mean (Private)")
        if private_present
        else pl.lit(None).cast(pl.Float64).alias("Mean (Private)")
    )

    joint_table = per_task.select(
        "model_name",
        *type_exprs,
        public_mean_expr,
        private_mean_expr,
    )

    return _PublicPrivateBuild(
        joint_table=joint_table,
        per_task=per_task,
        task_cols=task_cols,
        type_cols=type_cols,
        per_task_long=per_task_long,
        public_present=public_present,
        private_present=private_present,
    )


def _create_summary_table_mean_public_private(
    pl_df: pl.DataFrame,
    exclude_private_from_borda: bool = False,
) -> SummaryTable:
    """Create summary table that separates public and private task means.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``, ``score``,
            and ``is_public`` columns.
        exclude_private_from_borda: If True, calculate Borda rank using only public tasks.

    Returns:
        SummaryTable wrapping a frame with ``Mean (Public)`` (primary metric)
        and ``Mean (Private)`` plus the standard meta + type cols.
    """
    built = _build_public_private_joint(pl_df)
    if built is None:
        return _no_results_summary()

    # Long-form Borda from the per_task_long aggregate. When
    # ``exclude_private_from_borda`` is set and we have public tasks, filter
    # before ranking — matches the wide-form behaviour of using only
    # ``public_present`` as borda_cols. ``n_models`` matches wide-form's
    # ``pl.len()`` on per_task.
    borda_source = (
        built.per_task_long.filter(pl.col("is_public"))
        if exclude_private_from_borda and built.public_present
        else built.per_task_long
    )
    borda_df = _borda_rank_from_long(borda_source, n_models=built.per_task.height)

    joint_table = built.joint_table.join(borda_df, on="model_name", how="left")

    # Sort after attach: ~5-10% of rows get filtered as unknown models.
    joint_table = _attach_model_metadata(
        joint_table, task_names_key=tuple(sorted(built.task_cols))
    ).sort("Rank (Borda)")

    return SummaryTable(
        df=_order_summary_cols(
            joint_table,
            rank_col="Rank (Borda)",
            mean_cols=("Mean (Public)", "Mean (Private)"),
            type_cols=built.type_cols,
        ),
        rank_col="Rank (Borda)",
        # Primary IS the public mean — matches the prior API behaviour of
        # falling back to ``Mean (Public)`` when no ``Mean (Task)`` exists.
        primary_metric_col="Mean (Public)",
        task_type_mean_col=None,
        # ``mean_public_col=None`` because primary already IS the public mean.
        mean_public_col=None,
        mean_private_col="Mean (Private)",
    )


def _create_summary_table_mean_subset(
    pl_df: pl.DataFrame,
) -> SummaryTable:
    """Create summary table where each task-language subset is weighted equally.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            ``subset``, and ``score`` columns.

    Returns:
        SummaryTable wrapping a frame with ``Mean (Subset)`` as the primary
        metric plus the standard meta + type cols.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_summary()

    # Materialise the (model, task, subset) frame once — it feeds three
    # downstream pipelines (per-task pivot, per-model subset mean, per-subset
    # wide pivot for borda). Each downstream collect happens eagerly because
    # polars pivot isn't lazy; sharing the source avoids recomputing the
    # initial groupby thrice.
    per_subset_long = (
        pl_df.lazy()
        .group_by(["model_name", "task_name", "subset"])
        .agg(pl.col("score").mean())
        .collect()
    )

    per_task = (
        per_subset_long.group_by(["model_name", "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index="model_name", values="score")
    )
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return _no_results_summary()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_summary()

    type_exprs, type_cols = _get_means_per_types(task_cols)

    # Mean over all subset rows per model (each task-language subset weighted
    # equally) and borda over the per-(task, subset) long frame, partitioning
    # by the synthetic ``task::subset`` key so the rank-by-partition matches
    # the prior wide-form behaviour.
    overall_subset_mean = per_subset_long.group_by("model_name").agg(
        pl.col("score").mean().alias("Mean (Subset)")
    )
    per_subset_long_keyed = per_subset_long.with_columns(
        (pl.col("task_name") + "::" + pl.col("subset")).alias("_ts")
    )
    borda_df = _borda_rank_from_long(
        per_subset_long_keyed,
        partition_col="_ts",
        n_models=per_task.height,
    )

    joint_table = (
        per_task.select("model_name", *type_exprs)
        .join(overall_subset_mean, on="model_name", how="left")
        .join(borda_df, on="model_name", how="left")
    )

    # Sort after attach: ~5-10% of rows get filtered as unknown models.
    joint_table = _attach_model_metadata(
        joint_table, task_names_key=tuple(sorted(task_cols))
    ).sort("Mean (Subset)", descending=True, nulls_last=True)

    return SummaryTable(
        df=_order_summary_cols(
            joint_table,
            rank_col="Rank (Borda)",
            mean_cols=("Mean (Subset)",),
            type_cols=type_cols,
        ),
        rank_col="Rank (Borda)",
        primary_metric_col="Mean (Subset)",
        task_type_mean_col=None,
    )


def _create_summary_table_mean_task_type(
    pl_df: pl.DataFrame,
    mean_column_name: str = "Mean (TaskType)",
    sort_by: str | None = None,
    *,
    pivot: tuple[pl.DataFrame, list[str]] | None = None,
) -> SummaryTable:
    """Create summary table where the overall mean is the mean of per-task-type means.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.
        mean_column_name: Name for the mean-by-task-type column. Defaults to "Mean (TaskType)".
        sort_by: Column to sort the rows by (and to populate ``Rank``). When
            ``None`` falls back to ``mean_column_name`` (historical behaviour).
            Pass a non-None value when the benchmark wants to rank by a
            column that differs from its primary mean column.
        pivot: Pre-computed (per_task, task_cols) from :func:`_build_per_task_pivot`.
            Shared with the per-task table builder so the wide pivot is only
            computed once per long frame.

    Returns:
        SummaryTable with ``mean_column_name`` as both ``primary_metric_col``
        and ``task_type_mean_col`` (they hold the same value here). Rank is
        the sort-order rank by the configured ``sort_by``.
    """
    if pivot is None:
        pivot = _build_per_task_pivot(pl_df)
    if pivot is None:
        return _no_results_summary()
    per_task, task_cols = pivot

    joint_table, type_cols = _build_joint_with_type_means_and_borda(
        per_task,
        task_cols,
        pl_df,
        task_type_mean_alias=mean_column_name,
    )

    # Attach metadata before sort+rank: the 1-indexed ``Rank`` reflects only
    # known models (matches the previous behaviour after the inner-join
    # filter dropped unknowns post-sort) and we don't waste sort work on rows
    # that get filtered.
    sort_col = sort_by or mean_column_name
    joint_table = (
        _attach_model_metadata(joint_table, task_names_key=tuple(sorted(task_cols)))
        .sort(sort_col, descending=True, nulls_last=True)
        .with_columns((pl.int_range(0, pl.len()) + 1).cast(pl.Int64).alias("Rank"))
    )

    return SummaryTable(
        df=_order_summary_cols(
            joint_table,
            rank_col="Rank",
            mean_cols=(mean_column_name,),
            type_cols=type_cols,
            extra_trailing=("Rank (Borda)",),
        ),
        rank_col="Rank",
        primary_metric_col=mean_column_name,
        task_type_mean_col=mean_column_name,
    )
