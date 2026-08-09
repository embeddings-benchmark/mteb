"""Tests for the paired task-bootstrap rank-stability report."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mteb.benchmarks.rank_stability import bootstrap_rank_stability

N_BOOT = 2000
SEED = 12345


def _frame(scores: dict[str, list[float]], n_tasks: int | None = None) -> pd.DataFrame:
    """Wide frame in `BenchmarkResults.to_dataframe(format="wide")` shape."""
    n = n_tasks if n_tasks is not None else len(next(iter(scores.values())))
    return pd.DataFrame(scores, index=[f"Task{i}" for i in range(n)]).T


def test_report_states_its_own_sample_sizes() -> None:
    """A report that does not say what it measured over cannot be checked.

    Guards against the failure mode where an empty or truncated input silently
    produces a clean-looking report.
    """
    rng = np.random.default_rng(SEED)
    df = _frame(
        {
            "model/a": list(rng.normal(0.5, 0.1, 40)),
            "model/b": list(rng.normal(0.5, 0.1, 40)),
        }
    )
    report = bootstrap_rank_stability(df, n_boot=N_BOOT, seed=SEED)

    assert report.n_models == 2
    assert report.n_tasks == 40
    assert report.n_boot == N_BOOT
    assert report.dropped_models == ()
    assert report.dropped_tasks == ()
    assert len(report.summary) == 2
    assert report.win_probability.shape == (2, 2)


def test_mean_ci_matches_the_analytic_interval() -> None:
    """Known-value rung: the bootstrap CI must reproduce an interval derived
    independently of the bootstrap.

    For a mean over n independent tasks the 95% interval is
    ``mean +/- 1.96 * sd / sqrt(n)``. That value is computed here from the
    normal quantile and the sample sd, never from the resampler under test.
    """
    n_tasks = 2000
    rng = np.random.default_rng(SEED)
    scores = rng.normal(0.5, 0.1, n_tasks)
    df = _frame({"model/a": list(scores)})

    report = bootstrap_rank_stability(df, n_boot=N_BOOT, seed=SEED)
    row = report.summary.iloc[0]

    analytic_half_width = 1.959963984540054 * scores.std(ddof=1) / np.sqrt(n_tasks)
    bootstrap_half_width = (row["ci_high"] - row["ci_low"]) / 2

    assert row["mean"] == pytest.approx(scores.mean(), abs=1e-12)
    assert bootstrap_half_width == pytest.approx(analytic_half_width, rel=0.10)


def test_identical_models_are_reported_as_indistinguishable() -> None:
    """Negative control: no difference in the input, no difference in the output.

    Two models with the same score on every task must land at a win probability
    of exactly 0.5 and share the whole rank range.
    """
    rng = np.random.default_rng(SEED)
    shared = list(rng.normal(0.5, 0.15, 60))
    df = _frame({"model/a": shared, "model/b": list(shared)})

    report = bootstrap_rank_stability(df, n_boot=N_BOOT, seed=SEED)

    assert report.win_probability.loc["model/a", "model/b"] == pytest.approx(0.5)
    assert report.win_probability.loc["model/b", "model/a"] == pytest.approx(0.5)
    assert report.distinguishable_pairs() == []
    # Exact ties share the top rank under the min-rank convention; the signal
    # that they are not separated lives in the pairwise probability above.
    for model in ("model/a", "model/b"):
        assert report.summary.loc[model, "rank"] == 1


def test_uniform_offset_is_decisive_even_though_the_marginal_cis_overlap() -> None:
    """The reason the pairwise matrix exists rather than eyeballing the CIs.

    Model A beats model B on every single task by the same margin, so the paired
    comparison is decisive. The two marginal intervals still overlap almost
    entirely, because across-task spread dwarfs the offset. Reading
    "overlapping CIs" as "tied" would be wrong here, and this test pins both
    halves of that claim.
    """
    rng = np.random.default_rng(SEED)
    base = rng.normal(0.5, 0.2, 100)
    df = _frame({"model/a": list(base + 0.02), "model/b": list(base)})

    report = bootstrap_rank_stability(df, n_boot=N_BOOT, seed=SEED)
    a = report.summary.loc["model/a"]
    b = report.summary.loc["model/b"]

    # the marginal intervals overlap ...
    assert a["ci_low"] < b["ci_high"]
    assert b["ci_low"] < a["ci_high"]
    # ... and the paired comparison is still unanimous.
    assert report.win_probability.loc["model/a", "model/b"] == 1.0
    assert report.distinguishable_pairs() == [("model/a", "model/b", 1.0)]
    assert a["rank_ci_low"] == a["rank_ci_high"] == 1
    assert b["rank_ci_low"] == b["rank_ci_high"] == 2


def test_noise_only_difference_is_not_called_a_win() -> None:
    """Second negative control: a difference drawn from noise must not clear
    the distinguishability threshold."""
    rng = np.random.default_rng(SEED)
    df = _frame(
        {
            "model/a": list(rng.normal(0.5, 0.2, 80)),
            "model/b": list(rng.normal(0.5, 0.2, 80)),
        }
    )

    report = bootstrap_rank_stability(df, n_boot=N_BOOT, seed=SEED)

    assert report.distinguishable_pairs(threshold=0.95) == []
    # and the instability shows up in the ranks: both models reach both places
    for model in ("model/a", "model/b"):
        row = report.summary.loc[model]
        assert row["rank_ci_low"] == 1
        assert row["rank_ci_high"] == 2


def test_interval_narrows_as_the_task_count_grows() -> None:
    rng = np.random.default_rng(SEED)
    widths = []
    for n_tasks in (25, 100, 400):
        df = _frame({"model/a": list(rng.normal(0.5, 0.1, n_tasks))})
        row = bootstrap_rank_stability(df, n_boot=N_BOOT, seed=SEED).summary.iloc[0]
        widths.append(row["ci_high"] - row["ci_low"])

    assert widths[0] > widths[1] > widths[2]


def test_same_seed_reproduces_the_report_and_a_different_seed_does_not() -> None:
    rng = np.random.default_rng(SEED)
    df = _frame(
        {
            "model/a": list(rng.normal(0.5, 0.1, 50)),
            "model/b": list(rng.normal(0.5, 0.1, 50)),
        }
    )

    first = bootstrap_rank_stability(df, n_boot=500, seed=1).summary
    again = bootstrap_rank_stability(df, n_boot=500, seed=1).summary
    other = bootstrap_rank_stability(df, n_boot=500, seed=2).summary

    pd.testing.assert_frame_equal(first, again)
    assert not np.allclose(first["ci_low"].to_numpy(), other["ci_low"].to_numpy())


def test_win_probabilities_are_antisymmetric() -> None:
    rng = np.random.default_rng(SEED)
    df = _frame({f"model/{i}": list(rng.normal(0.5, 0.1, 40)) for i in range(4)})

    win = bootstrap_rank_stability(df, n_boot=N_BOOT, seed=SEED).win_probability
    values = win.to_numpy()

    assert np.allclose(np.diag(values), 0.5)
    assert np.allclose(values + values.T, 1.0)


def test_models_with_a_missing_task_are_dropped_and_named() -> None:
    """Default policy follows the leaderboard: a model missing a task has no
    `Mean (Task)` there either, so it leaves rather than shrinking the suite."""
    df = _frame(
        {
            "model/a": [0.1, 0.2, 0.3, 0.4],
            "model/b": [0.2, 0.3, 0.4, 0.5],
            "model/incomplete": [0.2, np.nan, 0.1, 0.5],
        }
    )

    report = bootstrap_rank_stability(df, n_boot=200, seed=SEED)

    assert report.dropped_models == ("model/incomplete",)
    assert report.dropped_tasks == ()
    assert report.n_models == 2
    assert report.n_tasks == 4


def test_drop_tasks_policy_keeps_every_model_and_names_the_cost() -> None:
    df = _frame(
        {
            "model/a": [0.1, 0.2, 0.3, 0.4],
            "model/b": [0.2, np.nan, 0.1, 0.5],
        }
    )

    report = bootstrap_rank_stability(
        df, n_boot=200, seed=SEED, on_missing="drop_tasks"
    )

    assert report.dropped_models == ()
    assert report.dropped_tasks == ("Task1",)
    assert report.n_models == 2
    assert report.n_tasks == 3


def test_a_task_no_model_ran_is_dropped_under_either_policy() -> None:
    df = _frame(
        {
            "model/a": [0.1, np.nan, 0.3, 0.4],
            "model/b": [0.2, np.nan, 0.1, 0.5],
        }
    )

    for policy in ("drop_models", "drop_tasks"):
        report = bootstrap_rank_stability(df, n_boot=200, seed=SEED, on_missing=policy)
        assert report.dropped_tasks == ("Task1",)
        assert report.n_models == 2
        assert report.n_tasks == 3


def test_unknown_missing_policy_raises() -> None:
    df = _frame({"model/a": [0.1, 0.2, 0.3], "model/b": [0.2, 0.3, 0.4]})

    with pytest.raises(ValueError, match="on_missing"):
        bootstrap_rank_stability(df, n_boot=200, seed=SEED, on_missing="ignore")


def test_ranks_follow_the_scores() -> None:
    df = _frame(
        {
            "model/low": list(np.full(30, 0.10)),
            "model/high": list(np.full(30, 0.90)),
            "model/mid": list(np.full(30, 0.50)),
        }
    )

    summary = bootstrap_rank_stability(df, n_boot=200, seed=SEED).summary

    assert list(summary.index) == ["model/high", "model/mid", "model/low"]
    assert list(summary["rank"]) == [1, 2, 3]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_boot": 0}, "n_boot"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"confidence_level": 0.0}, "confidence_level"),
    ],
)
def test_invalid_arguments_raise(kwargs: dict, match: str) -> None:
    df = _frame({"model/a": [0.1, 0.2, 0.3], "model/b": [0.2, 0.3, 0.4]})

    with pytest.raises(ValueError, match=match):
        bootstrap_rank_stability(df, seed=SEED, **kwargs)


def test_too_few_tasks_raises_rather_than_reporting_a_zero_width_interval() -> None:
    df = _frame({"model/a": [0.1], "model/b": [0.2]})

    with pytest.raises(ValueError, match="at least 2 tasks"):
        bootstrap_rank_stability(df, n_boot=200, seed=SEED)


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="no models"):
        bootstrap_rank_stability(pd.DataFrame(), n_boot=200, seed=SEED)
