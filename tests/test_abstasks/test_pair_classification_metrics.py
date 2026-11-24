import numpy as np
import pytest

from mteb.mocks.mock_tasks import MockPairClassificationTask


@pytest.mark.parametrize("high_score_more_similar", [True, False])
def test_constant_scores_do_not_depend_on_row_order(high_score_more_similar: bool):
    # a threshold can't separate pairs with the same score, so with a constant scorer
    # the metrics must not depend on whether positives are listed first or last
    task = MockPairClassificationTask()
    scores = [0.5, 0.5, 0.5, 0.5]

    positives_first = task._compute_metrics_values(
        scores, np.array([1, 1, 0, 0]), high_score_more_similar
    )
    positives_last = task._compute_metrics_values(
        scores, np.array([0, 0, 1, 1]), high_score_more_similar
    )

    assert positives_first == positives_last
    assert positives_first["accuracy"] < 1.0
    assert positives_first["f1"] < 1.0


@pytest.mark.parametrize("high_score_more_similar", [True, False])
def test_partial_ties_are_not_split(high_score_more_similar: bool):
    scores = [0.9, 0.5, 0.5, 0.1]
    if not high_score_more_similar:
        scores = [-s for s in scores]
    task = MockPairClassificationTask()

    # the positive and the negative with score 0.5 can't be told apart by any threshold
    tied_positive_first = task._compute_metrics_values(
        scores, np.array([1, 1, 0, 0]), high_score_more_similar
    )
    tied_negative_first = task._compute_metrics_values(
        scores, np.array([1, 0, 1, 0]), high_score_more_similar
    )

    assert tied_positive_first == tied_negative_first
    assert tied_positive_first["accuracy"] == pytest.approx(0.75)
    assert tied_positive_first["f1"] == pytest.approx(0.8)


def test_scores_without_ties_are_unchanged():
    task = MockPairClassificationTask()
    metrics = task._compute_metrics_values(
        [0.9, 0.8, 0.3, 0.1], np.array([1, 1, 0, 0]), True
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0


@pytest.mark.parametrize("high_score_more_similar", [True, False])
def test_all_positive_threshold_is_considered(high_score_more_similar: bool):
    # every pair is positive, so predicting all of them as positive is the best threshold
    scores = [0.9, 0.5, 0.1]
    if not high_score_more_similar:
        scores = [-s for s in scores]
    task = MockPairClassificationTask()
    metrics = task._compute_metrics_values(
        scores, np.array([1, 1, 1]), high_score_more_similar
    )
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["recall"] == 1.0


def test_all_negative_threshold_is_considered():
    # the lowest scored pair is the only positive, so predicting every pair as negative
    # gives the best accuracy, and predicting every pair as positive gives the best f1
    task = MockPairClassificationTask()
    metrics = task._compute_metrics_values(
        [0.9, 0.8, 0.7, 0.1], np.array([0, 0, 0, 1]), True
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(0.4)


def test_constant_scores_use_the_all_positive_threshold():
    task = MockPairClassificationTask()
    metrics = task._compute_metrics_values(
        [0.5, 0.5, 0.5, 0.5], np.array([1, 1, 1, 0]), True
    )
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(6 / 7)
