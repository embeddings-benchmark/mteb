"""Tests for mteb.api.bradley_terry (BT score aggregation of per-task scores)."""

from __future__ import annotations

import math

import pytest

from mteb.api.bradley_terry import BT_SCORE_BASE, compute_bt_score

TASKS = ["t1", "t2", "t3", "t4"]


def _scores(**rows: dict[str, float]) -> dict[str, dict[str, float]]:
    return rows


def test_strict_dominance_orders_models():
    res = compute_bt_score(
        _scores(
            a=dict.fromkeys(TASKS, 3.0),
            b=dict.fromkeys(TASKS, 2.0),
            c=dict.fromkeys(TASKS, 1.0),
        ),
        TASKS,
        n_boot=0,
    )
    assert res["a"].score > res["b"].score > res["c"].score
    assert all(r.low is None and r.high is None for r in res.values())


def test_identical_models_tie_and_center_on_base():
    res = compute_bt_score(
        _scores(a=dict.fromkeys(TASKS, 0.5), b=dict.fromkeys(TASKS, 0.5)),
        TASKS,
        n_boot=0,
    )
    assert res["a"].score == pytest.approx(res["b"].score)
    assert res["a"].score == pytest.approx(BT_SCORE_BASE)


def test_mean_elo_is_base():
    res = compute_bt_score(
        _scores(
            a={"t1": 3, "t2": 1, "t3": 2, "t4": 5},
            b={"t1": 2, "t2": 3, "t3": 1, "t4": 4},
            c={"t1": 1, "t2": 2, "t3": 3, "t4": 1},
        ),
        TASKS,
        n_boot=0,
    )
    assert sum(r.score for r in res.values()) / 3 == pytest.approx(BT_SCORE_BASE)


def test_missing_task_counts_as_loss():
    full = dict.fromkeys(TASKS, 0.9)
    partial = {"t1": 0.9}
    other = dict.fromkeys(TASKS, 0.5)
    res = compute_bt_score(
        _scores(full=full, partial=partial, other=other), TASKS, n_boot=0
    )
    # partial equals `full` on its only task but loses the rest to `other`
    assert res["partial"].score < res["other"].score < res["full"].score


def test_skipping_tasks_does_not_inflate():
    strong = dict.fromkeys(TASKS, 1.0)
    weak = dict.fromkeys(TASKS, 0.1)
    cherry = {"t1": 2.0}  # best score on the one task it ran, nothing elsewhere
    res = compute_bt_score(
        _scores(strong=strong, weak=weak, cherry=cherry), TASKS, n_boot=0
    )
    assert res["cherry"].score < res["strong"].score


def test_invariant_to_monotone_rescale_of_a_task():
    base = {
        "a": {"t1": 0.2, "t2": 70.0, "t3": 0.5, "t4": 3.0},
        "b": {"t1": 0.4, "t2": 60.0, "t3": 0.1, "t4": 2.0},
        "c": {"t1": 0.3, "t2": 65.0, "t3": 0.9, "t4": 1.0},
    }
    scaled = {k: {**v, "t2": math.log(v["t2"]) * 100} for k, v in base.items()}
    r1 = compute_bt_score(base, TASKS, n_boot=0)
    r2 = compute_bt_score(scaled, TASKS, n_boot=0)
    for k in base:
        assert r1[k].score == pytest.approx(r2[k].score)


def test_ci_contains_point_estimate_and_is_deterministic():
    data = {
        "a": {"t1": 3, "t2": 1, "t3": 2, "t4": 5},
        "b": {"t1": 2, "t2": 3, "t3": 1, "t4": 4},
        "c": {"t1": 1, "t2": 2, "t3": 3, "t4": 1},
    }
    r1 = compute_bt_score(data, TASKS, n_boot=30, seed=7)
    r2 = compute_bt_score(data, TASKS, n_boot=30, seed=7)
    assert r1 == r2
    for r in r1.values():
        assert r.low is not None and r.high is not None
        assert r.low <= r.score <= r.high


def test_degenerate_inputs_return_empty():
    assert compute_bt_score({"a": {"t1": 1.0}}, TASKS) == {}
    assert compute_bt_score({"a": {"t1": 1.0}, "b": {"t1": 2.0}}, []) == {}


def test_nan_scores_are_missing():
    res = compute_bt_score(
        _scores(a={"t1": float("nan")}, b={"t1": 0.1}), ["t1"], n_boot=0
    )
    assert res["b"].score > res["a"].score
