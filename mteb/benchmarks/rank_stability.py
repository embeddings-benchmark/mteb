"""Uncertainty on a benchmark's aggregate score, and on the ranking built from it.

A benchmark mean is an average over the tasks that happen to be in the suite. Two
models a few thousandths apart on that mean can swap places when the suite gains
or loses a handful of tasks, and the leaderboard gives a reader no way to see it.

This module resamples the *tasks* — the unit the aggregate averages over — and
reports what survives:

- a percentile confidence interval on each model's mean;
- the range of ranks a model occupies across resamples;
- for every pair of models, how often one beats the other.

The resample is **paired**: one draw of tasks is applied to every model, so the
cross-model correlation that makes comparisons much sharper than the marginal
intervals is preserved. Two models whose intervals overlap almost entirely can
still be separated on every resample, which is why the pairwise matrix is the
thing to read for a comparison, not the overlap of two intervals.

What this measures is sensitivity of the aggregate to benchmark composition. It
is not the sampling error of the individual task scores, which would need the
per-task sample sizes and lives at the task level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["RankStabilityReport", "bootstrap_rank_stability"]

_MIN_TASKS = 2


@dataclass(frozen=True, slots=True)
class RankStabilityReport:
    """Result of [bootstrap_rank_stability][mteb.benchmarks.rank_stability.bootstrap_rank_stability].

    Attributes:
        summary: One row per model, sorted best first. Columns: `mean` (the
            aggregate over the tasks used), `ci_low` / `ci_high` (percentile
            interval on that mean — the spread of the aggregate under a
            different draw of tasks; two models are *not* to be compared by
            whether these overlap, `win_probability` is the number for that),
            `rank` (1 = best, ties share the lower number), and `rank_ci_low` /
            `rank_ci_high` (the interval of ranks the model occupies across
            resamples).
        win_probability: Square frame; entry `(i, j)` is the share of resamples
            in which model `i` scores above model `j`, ties counted as a half.
            The diagonal is 0.5 and the matrix sums to 1 with its transpose.
        n_models: Models in the report.
        n_tasks: Tasks the report was computed over.
        n_boot: Resamples drawn.
        confidence_level: Level of the reported intervals.
        seed: Seed the resampler ran with; the report is reproducible from it.
        dropped_models: Models excluded because they were missing at least one
            task, under `on_missing="drop_models"`.
        dropped_tasks: Tasks excluded — ones no model ran, plus the rest of the
            incomplete ones under `on_missing="drop_tasks"`.

    Both dropped lists are named rather than counted so a caller can see what
    the pairing cost.
    """

    summary: pd.DataFrame
    win_probability: pd.DataFrame
    n_models: int
    n_tasks: int
    n_boot: int
    confidence_level: float
    seed: int
    dropped_models: tuple[str, ...]
    dropped_tasks: tuple[str, ...]

    def distinguishable_pairs(
        self, threshold: float = 0.95
    ) -> list[tuple[str, str, float]]:
        """Pairs one model wins at least `threshold` of the time.

        Args:
            threshold: Win share a pair must reach to be listed.

        Returns:
            `(winner, loser, win_probability)` tuples, most decisive first.
            Anything absent from this list is a pair the benchmark does not
            separate at this level, however far apart the two means look.
        """
        models = list(self.win_probability.index)
        values = self.win_probability.to_numpy()
        pairs = [
            (models[i], models[j], float(values[i, j]))
            for i in range(len(models))
            for j in range(len(models))
            if i != j and values[i, j] >= threshold
        ]
        return sorted(pairs, key=lambda pair: -pair[2])


def _validate(
    n_boot: int, confidence_level: float, n_models: int, n_tasks: int
) -> None:
    """Reject arguments that would produce a confident-looking empty measurement."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be at least 1, got {n_boot}")
    if not 0 < confidence_level < 1:
        raise ValueError(
            f"confidence_level must lie strictly between 0 and 1, got {confidence_level}"
        )
    if n_models < 1:
        raise ValueError("the score frame contains no models")
    if n_tasks < _MIN_TASKS:
        raise ValueError(
            f"rank stability needs at least 2 tasks to resample over, got {n_tasks}"
        )


def _bootstrap_means(scores: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Paired task-bootstrap of the per-model mean.

    Draws multinomial task counts once per resample and applies the same draw to
    every model — equivalent to sampling task indices with replacement, but it
    keeps the working set at `(n_models, n_boot)` instead of materialising an
    `(n_models, n_boot, n_tasks)` index expansion.

    Returns:
        Array of shape `(n_models, n_boot)`.
    """
    n_tasks = scores.shape[1]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n_tasks, np.full(n_tasks, 1 / n_tasks), size=n_boot)
    return scores @ counts.T / n_tasks


def _ranks_and_win_probabilities(
    boot_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-resample ranks and pairwise win shares from the bootstrap means.

    One model at a time rather than an `(n_models, n_models, n_boot)` tensor, so
    peak memory stays at one `(n_models, n_boot)` comparison for a leaderboard
    with hundreds of models.

    Returns:
        `(boot_ranks, win_probability)` of shapes `(n_models, n_boot)` and
        `(n_models, n_models)`.
    """
    n_models, n_boot = boot_means.shape
    boot_ranks = np.empty((n_models, n_boot), dtype=np.int64)
    win_probability = np.empty((n_models, n_models), dtype=float)
    for i in range(n_models):
        above = boot_means > boot_means[i]
        equal = boot_means == boot_means[i]
        # min-rank convention: 1 + how many models scored strictly higher.
        boot_ranks[i] = 1 + above.sum(axis=0)
        # ties split, so P(i beats j) + P(j beats i) == 1 exactly.
        win_probability[i] = (~above & ~equal).mean(axis=1) + 0.5 * equal.mean(axis=1)
    return boot_ranks, win_probability


def _summary_frame(
    models: list[str],
    values: np.ndarray,
    boot_means: np.ndarray,
    boot_ranks: np.ndarray,
    alpha: float,
) -> pd.DataFrame:
    """Per-model means, intervals and rank intervals, best first.

    Rank intervals take integer quantiles (`inverted_cdf`) because a rank
    halfway between two places is not a place a model can be shown in.
    """
    quantiles = [alpha, 1 - alpha]
    means = values.mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, quantiles, axis=1)
    rank_low, rank_high = np.quantile(
        boot_ranks, quantiles, axis=1, method="inverted_cdf"
    )
    return pd.DataFrame(
        {
            "mean": means,
            "ci_low": ci_low,
            "ci_high": ci_high,
            # min-rank, matching the per-resample ranks.
            "rank": 1 + (means[:, None] < means[None, :]).sum(axis=1),
            "rank_ci_low": rank_low.astype(np.int64),
            "rank_ci_high": rank_high.astype(np.int64),
        },
        index=pd.Index(models, name="model_name"),
    ).sort_values("mean", ascending=False, kind="stable")


def _make_complete(
    numeric: pd.DataFrame, on_missing: str
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Reduce to a rectangle every model ran, and say what that cost.

    A paired resample needs one common task set, so something has to give when
    a model is missing a task. `"drop_models"` follows the leaderboard, whose
    `Mean (Task)` is null for a model with a missing task — such a model is not
    ranked by the mean in the first place. `"drop_tasks"` keeps every model at
    the price of shrinking the benchmark. Either way both lists come back named.
    """
    if on_missing not in {"drop_models", "drop_tasks"}:
        raise ValueError(
            f'on_missing must be "drop_models" or "drop_tasks", got {on_missing!r}'
        )
    # A task nobody ran carries no information under either policy.
    complete = numeric.dropna(axis="columns", how="all")
    if on_missing == "drop_models":
        kept = complete.dropna(axis="index", how="any")
        dropped_models = tuple(
            str(model) for model in complete.index.difference(kept.index)
        )
    else:
        kept = complete.dropna(axis="columns", how="any")
        dropped_models = ()
    dropped_tasks = tuple(
        str(task) for task in numeric.columns.difference(kept.columns)
    )
    return kept, dropped_models, dropped_tasks


def bootstrap_rank_stability(
    scores: pd.DataFrame,
    *,
    n_boot: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
    on_missing: str = "drop_models",
) -> RankStabilityReport:
    """Confidence intervals and rank stability for a benchmark's aggregate score.

    Args:
        scores: Wide frame of scores, one row per model and one column per task
            — the shape
            [BenchmarkResults.to_dataframe][mteb.results.benchmark_results.BenchmarkResults.to_dataframe]
            returns with `format="wide"`.
        n_boot: Resamples to draw. The default is enough for the intervals;
            reading a win probability out at three decimals wants more.
        confidence_level: Level for the reported intervals.
        seed: Seed for the resampler. The report is reproducible from it.
        on_missing: How to reach the common task set the pairing needs.
            `"drop_models"` (default) keeps the benchmark whole and drops models
            with a missing task, matching the leaderboard's own `Mean (Task)`,
            which is null for exactly those models. `"drop_tasks"` keeps every
            model and drops the incomplete tasks instead.

    Returns:
        RankStabilityReport: intervals on the means, the range of ranks each
            model occupies, and the pairwise win probabilities.

    Raises:
        ValueError: If no models or fewer than two tasks survive the
            completeness filter, or if `n_boot`, `confidence_level` or
            `on_missing` are out of range.

    Examples:
        >>> import mteb
        >>> from mteb.benchmarks.rank_stability import bootstrap_rank_stability
        >>> benchmark = mteb.get_benchmark("MTEB(eng, v2)")  # doctest: +SKIP
        >>> wide = benchmark.load_results().to_dataframe(format="wide")  # doctest: +SKIP
        >>> report = bootstrap_rank_stability(wide)  # doctest: +SKIP
        >>> report.summary.head()  # doctest: +SKIP
        >>> report.distinguishable_pairs()  # doctest: +SKIP
    """
    if scores.empty:
        raise ValueError("the score frame contains no models")

    numeric = scores.apply(pd.to_numeric, errors="coerce")
    complete, dropped_models, dropped_tasks = _make_complete(numeric, on_missing)

    _validate(n_boot, confidence_level, len(complete.index), complete.shape[1])

    models = [str(model) for model in complete.index]
    values = complete.to_numpy(dtype=float)

    boot_means = _bootstrap_means(values, n_boot, seed)
    boot_ranks, win_probability = _ranks_and_win_probabilities(boot_means)

    return RankStabilityReport(
        summary=_summary_frame(
            models, values, boot_means, boot_ranks, (1 - confidence_level) / 2
        ),
        win_probability=pd.DataFrame(win_probability, index=models, columns=models),
        n_models=len(models),
        n_tasks=complete.shape[1],
        n_boot=n_boot,
        confidence_level=confidence_level,
        seed=seed,
        dropped_models=dropped_models,
        dropped_tasks=dropped_tasks,
    )
