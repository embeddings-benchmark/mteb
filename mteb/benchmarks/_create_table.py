from __future__ import annotations

import functools
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from mteb.get_tasks import _TASKS_REGISTRY
from mteb.models.get_model_meta import get_model_meta


@functools.lru_cache(maxsize=4096)
def _training_datasets_cached(model_name: str) -> frozenset[str] | None:
    """Memoized training datasets (with similar tasks) for a model.

    The similar-task graph traversal in ``ModelMeta.get_training_datasets()`` is
    expensive and depends only on the model, so cache it per model name here at the
    leaderboard layer (rather than polluting ``ModelMeta``). Both the summary's
    zero-shot column and ``_filter_models``' zero-shot check share this cache.
    """
    meta = get_model_meta(model_name)
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


def _no_results_frame() -> pl.DataFrame:
    """The placeholder frame returned when an empty selection would have no rows."""
    return pl.DataFrame({"No results": ["You can try relaxing your criteria"]})


def _skipna_false_mean(cols: list[str]) -> pl.Expr:
    """Row-wise mean that returns null if any of ``cols`` is null.

    Matches ``pd.DataFrame.mean(axis=1, skipna=False)`` semantics.
    """
    any_null = pl.any_horizontal([pl.col(c).is_null() for c in cols])
    return pl.when(any_null).then(None).otherwise(pl.mean_horizontal(cols))


def _get_borda_rank(wide: pl.DataFrame, score_cols: list[str]) -> pl.Series:
    """Borda rank for each row across ``score_cols``.

    Per-column rank (higher score → lower rank number) is converted to a borda count
    (``n - rank``), summed row-wise, and ranked again with ``method="min"``. Returns a
    polars ``Int64`` Series in the same row order as ``wide``.
    """
    n = wide.height
    return wide.select(
        pl.sum_horizontal(
            [n - pl.col(c).rank(method="average", descending=True) for c in score_cols]
        )
        .rank(method="min", descending=True)
        .cast(pl.Int64)
        .alias("rank")
    ).to_series()


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


def _format_max_tokens(max_tokens: float | None) -> float | None:
    if max_tokens is None or max_tokens == np.inf:
        return None
    return float(max_tokens)


def _get_embedding_size(embed_dim: int | list[int] | None) -> int | None:
    if embed_dim is None:
        return None
    if isinstance(embed_dim, int):
        return int(embed_dim)
    if isinstance(embed_dim, Sequence) and len(embed_dim) > 0:
        return int(max(embed_dim))
    return None


def _get_means_per_types(
    per_task: pl.DataFrame, task_cols: list[str]
) -> tuple[pl.DataFrame, list[str]]:
    """Compute per-task-type means in polars.

    Returns ``(wide_frame, type_cols)`` where ``wide_frame`` has ``model_name`` plus
    one column per task type (column names already passed through ``_split_on_capital``),
    and ``type_cols`` lists those column names in insertion order. Means are computed
    with ``skipna=False`` semantics (matches the prior pandas implementation).
    """
    task_names_per_type: dict[str, list[str]] = defaultdict(list)
    for task_name in task_cols:
        # Read from the registered class to skip instantiation (get_task() runs filter_languages()).
        task_type = _TASKS_REGISTRY[task_name].metadata.type
        task_names_per_type[task_type].append(task_name)

    type_cols = [_split_on_capital(t) for t in task_names_per_type]
    return (
        per_task.select(
            pl.col("model_name"),
            *(
                _skipna_false_mean(tasks).alias(_split_on_capital(task_type))
                for task_type, tasks in task_names_per_type.items()
            ),
        ),
        type_cols,
    )


def _attach_model_metadata(
    joint_table: pl.DataFrame,
    task_names_key: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Filter to models with valid metadata and attach the standard summary columns.

    Inner-joins meta columns (``Max Tokens``, ``Embedding Dimensions``, ``Total/Active
    Parameters (B)``, ``Release Date``) onto ``joint_table`` (which must have a
    ``model_name`` column), replaces ``model_name`` with a markdown-linked ``Model``
    column, and optionally adds a ``Zero-shot`` column when ``task_names_key`` is
    provided (None → -1 to mirror the previous ``.fillna(-1)``).
    """
    # Single pass over the models — each meta is touched once and accumulated into
    # parallel lists, then assembled into a polars frame. Avoids the previous N
    # separate comprehensions which each re-walked the metas list.
    names: list[str] = []
    max_tokens: list[float | None] = []
    embed_dims: list[int | None] = []
    total_params: list[float | None] = []
    active_params: list[float | None] = []
    release_dates: list[str | None] = []
    model_links: list[str | None] = []
    zero_shots: list[int] | None = [] if task_names_key is not None else None

    for name in joint_table.get_column("model_name").to_list():
        m = get_model_meta(name)
        if m is None:
            continue
        names.append(name)
        max_tokens.append(_format_max_tokens(m.max_tokens))
        embed_dims.append(_get_embedding_size(m.embed_dim))
        total_params.append(_format_n_parameters(m.n_parameters))
        active_params.append(_format_n_parameters(m.n_active_parameters))
        release_dates.append(str(m.release_date) if m.release_date else None)
        model_links.append(m.reference)
        if zero_shots is not None:
            z = _zero_shot_pct_cached(m.name, task_names_key)
            zero_shots.append(-1 if z is None else z)

    rows: dict[str, list] = {
        "model_name": names,
        "Max Tokens": max_tokens,
        "Embedding Dimensions": embed_dims,
        "Total Parameters (B)": total_params,
        "Active Parameters (B)": active_params,
        "Release Date": release_dates,
        "_model_link": model_links,
    }
    if zero_shots is not None:
        rows["Zero-shot"] = zero_shots
    meta_df = pl.DataFrame(rows)
    return (
        joint_table.join(meta_df, on="model_name", how="inner")
        .with_columns(
            pl.col("model_name").str.split("/").list.last().alias("_short_name")
        )
        .with_columns(
            pl.when(pl.col("_model_link").is_not_null())
            .then("[" + pl.col("_short_name") + "](" + pl.col("_model_link") + ")")
            .otherwise(pl.col("_short_name"))
            .alias("Model")
        )
        .drop(["_model_link", "_short_name", "model_name"])
    )


def _create_summary_table_from_benchmark_results(
    pl_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create summary table from a long polars pre-aggregation frame.

    Stays in polars throughout (aggregation, pivot, type-means, borda, model metadata,
    markdown link, sort, rename); converts to pandas only at the return boundary so the
    leaderboard's pandas Styler can consume it.

    Returns a DataFrame with one row per model containing summary statistics
    and task type averages.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    per_task = (
        pl_df.group_by(["model_name", "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index="model_name", values="score")
    )
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return _no_results_frame()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    mean_per_type, type_cols = _get_means_per_types(per_task, task_cols)

    joint_table = (
        mean_per_type.with_columns(
            _skipna_false_mean(type_cols).alias("Mean (TaskType)"),
        )
        .join(
            per_task.select(
                "model_name",
                _skipna_false_mean(task_cols).alias("Mean (Task)"),
            ),
            on="model_name",
            how="left",
        )
        .with_columns(_get_borda_rank(per_task, task_cols).alias("Rank (Borda)"))
        .sort("Rank (Borda)")
    )

    joint_table = _attach_model_metadata(
        joint_table, task_names_key=tuple(sorted(task_cols))
    )

    final_cols = [
        "Rank (Borda)",
        "Model",
        "Zero-shot",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        "Mean (Task)",
        "Mean (TaskType)",
        *type_cols,
        "Release Date",
    ]
    return joint_table.select([c for c in final_cols if c in joint_table.columns])


def _create_per_task_table_from_benchmark_results(
    pl_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create per-task table from a long polars pre-aggregation frame.

    All aggregation, ranking, and sorting runs in polars; the result is converted to
    pandas only at the return boundary (the leaderboard's Styler is pandas-based).

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.

    Returns:
        DataFrame with per-task scores, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    per_task = (
        pl_df.group_by(["model_name", "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index="model_name", values="score")
    )
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return _no_results_frame()

    # Drop models whose task scores are all null.
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    per_task = (
        per_task.with_columns(_get_borda_rank(per_task, task_cols).alias("_borda"))
        .sort("_borda")
        .drop("_borda")
        .with_columns(pl.col("model_name").str.split("/").list.last().alias("Model"))
        .drop("model_name")
        .select(["Model", *task_cols])
    )
    return per_task


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

    lang_df = (
        pl_df.explode("language")
        .drop_nulls("language")
        .group_by(["model_name", "language"])
        .agg(pl.col("score").mean())
    )
    if language_view != "all":
        lang_df = lang_df.filter(pl.col("language").is_in(language_view))
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

    per_language = (
        per_language.with_columns(
            _get_borda_rank(per_language, lang_cols).alias("_borda")
        )
        .sort("_borda")
        .drop("_borda")
        .with_columns(pl.col("model_name").str.split("/").list.last().alias("Model"))
        .drop("model_name")
        .select(["Model", *lang_cols])
    )
    return per_language


def _create_summary_table_mean_public_private(
    pl_df: pl.DataFrame,
    exclude_private_from_borda: bool = False,
) -> pl.DataFrame:
    """Create summary table that separates public and private task means.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``, ``score``,
            and ``is_public`` columns.
        exclude_private_from_borda: If True, calculate Borda rank using only public tasks.

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    per_task_long = pl_df.group_by(["model_name", "task_name"]).agg(
        pl.col("score").mean(),
        pl.col("is_public").first(),
    )
    public_tasks = (
        per_task_long.filter(pl.col("is_public"))
        .get_column("task_name")
        .unique()
        .to_list()
    )
    private_tasks = (
        per_task_long.filter(~pl.col("is_public"))
        .get_column("task_name")
        .unique()
        .to_list()
    )
    per_task = per_task_long.pivot(on="task_name", index="model_name", values="score")
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return _no_results_frame()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    mean_per_type, type_cols = _get_means_per_types(per_task, task_cols)

    public_present = [c for c in public_tasks if c in task_cols]
    private_present = [c for c in private_tasks if c in task_cols]
    borda_cols = (
        public_present if exclude_private_from_borda and public_present else task_cols
    )

    joint_table = (
        mean_per_type.join(
            per_task.select(
                "model_name",
                (
                    _skipna_false_mean(public_present).alias("Mean (Public)")
                    if public_present
                    else pl.lit(None).cast(pl.Float64).alias("Mean (Public)")
                ),
                (
                    _skipna_false_mean(private_present).alias("Mean (Private)")
                    if private_present
                    else pl.lit(None).cast(pl.Float64).alias("Mean (Private)")
                ),
            ),
            on="model_name",
            how="left",
        )
        .with_columns(_get_borda_rank(per_task, borda_cols).alias("Rank (Borda)"))
        .sort("Rank (Borda)")
    )

    joint_table = _attach_model_metadata(joint_table)

    final_cols = [
        "Rank (Borda)",
        "Model",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        "Mean (Public)",
        "Mean (Private)",
        *type_cols,
        "Release Date",
    ]
    return joint_table.select([c for c in final_cols if c in joint_table.columns])


def _create_summary_table_mean_subset(
    pl_df: pl.DataFrame,
) -> pl.DataFrame:
    """Create summary table where each task-language subset is weighted equally.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            ``subset``, and ``score`` columns.

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    # Per-task mean (for per-type aggregation) and per-(task,subset) mean (for borda).
    per_subset_long = pl_df.group_by(["model_name", "task_name", "subset"]).agg(
        pl.col("score").mean()
    )
    per_task = (
        per_subset_long.group_by(["model_name", "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index="model_name", values="score")
    )
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return _no_results_frame()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    mean_per_type, type_cols = _get_means_per_types(per_task, task_cols)

    # Mean over all subset rows per model (each task-language subset weighted equally).
    overall_subset_mean = per_subset_long.group_by("model_name").agg(
        pl.col("score").mean().alias("Mean (Subset)")
    )
    # Borda over per-(task, subset) columns. Pivot creates "task__subset"-shaped names,
    # but the exact names don't matter — we only need the score columns for ranking.
    per_subset_wide = per_subset_long.with_columns(
        (pl.col("task_name") + "::" + pl.col("subset")).alias("_ts")
    ).pivot(on="_ts", index="model_name", values="score")
    subset_cols = [c for c in per_subset_wide.columns if c != "model_name"]

    joint_table = (
        mean_per_type.join(overall_subset_mean, on="model_name", how="left")
        .join(
            per_subset_wide.select(
                "model_name",
                _get_borda_rank(per_subset_wide, subset_cols).alias("Rank (Borda)"),
            ),
            on="model_name",
            how="left",
        )
        .sort("Mean (Subset)", descending=True, nulls_last=True)
    )

    joint_table = _attach_model_metadata(
        joint_table, task_names_key=tuple(sorted(task_cols))
    )

    final_cols = [
        "Rank (Borda)",
        "Model",
        "Zero-shot",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        "Mean (Subset)",
        *type_cols,
        "Release Date",
    ]
    return joint_table.select([c for c in final_cols if c in joint_table.columns])


def _create_summary_table_mean_task_type(
    pl_df: pl.DataFrame, mean_column_name: str = "Mean (TaskType)"
) -> pl.DataFrame:
    """Create summary table where the overall mean is the mean of per-task-type means.

    Args:
        pl_df: Long polars frame with at least ``model_name``, ``task_name``,
            and ``score`` columns.
        mean_column_name: Name for the mean-by-task-type column. Defaults to "Mean (TaskType)".

    Returns:
        DataFrame with model summaries, ready for styling in the leaderboard.
    """
    if pl_df.is_empty() or "model_name" not in pl_df.columns:
        return _no_results_frame()

    per_task = (
        pl_df.group_by(["model_name", "task_name"])
        .agg(pl.col("score").mean())
        .pivot(on="task_name", index="model_name", values="score")
    )
    task_cols = [c for c in per_task.columns if c != "model_name"]
    if not task_cols:
        return _no_results_frame()
    per_task = per_task.filter(
        pl.any_horizontal([pl.col(c).is_not_null() for c in task_cols])
    )
    if per_task.is_empty():
        return _no_results_frame()

    mean_per_type, type_cols = _get_means_per_types(per_task, task_cols)

    joint_table = (
        mean_per_type.with_columns(
            _skipna_false_mean(type_cols).alias(mean_column_name),
        )
        .sort(mean_column_name, descending=True, nulls_last=True)
        .with_columns(
            _get_borda_rank(per_task, task_cols).alias("Rank (Borda)"),
            (pl.int_range(0, pl.len()) + 1).cast(pl.Int64).alias("Rank"),
        )
    )

    joint_table = _attach_model_metadata(
        joint_table, task_names_key=tuple(sorted(task_cols))
    )

    # Renames specific to mean-task-type variants (Vidore/MIEB).
    renames: dict[str, str] = {}
    if "Any Any Multilingual Retrieval" in joint_table.columns:
        renames["Any Any Multilingual Retrieval"] = "Multilingual Retrieval"
    if "Any Any Retrieval" in joint_table.columns:
        renames["Any Any Retrieval"] = "Retrieval"
    if renames:
        joint_table = joint_table.rename(renames)
        type_cols = [renames.get(c, c) for c in type_cols]

    final_cols = [
        "Rank",
        "Model",
        "Zero-shot",
        "Active Parameters (B)",
        "Total Parameters (B)",
        "Embedding Dimensions",
        "Max Tokens",
        mean_column_name,
        *type_cols,
        "Rank (Borda)",
        "Release Date",
    ]
    return joint_table.select([c for c in final_cols if c in joint_table.columns])
