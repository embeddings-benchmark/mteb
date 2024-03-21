from __future__ import annotations

import pytest

from mteb.evaluation.evaluators import PairClassificationEvaluator

TOL = 0.0001


class TestPairClassificationEvaluator:
    def test_accuracy(self):
        scores = [6.12, 5.39, 5.28, 5.94, 6.34, 6.47, 7.88, 6.62, 8.04, 5.9]
        labels = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        high_score_more_similar = True
        acc, acc_threshold = PairClassificationEvaluator.find_best_acc_and_threshold(
            scores, labels, high_score_more_similar
        )
        assert acc == pytest.approx(0.9, TOL)
        assert acc_threshold == pytest.approx(7.95999, TOL)

    def test_f1(self):
        scores = [6.12, 5.39, 5.28, 5.94, 6.34, 6.47, 7.88, 6.62, 8.04, 5.9]
        labels = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        high_score_more_similar = True

        f1, precision, recall, f1_threshold = (
            PairClassificationEvaluator.find_best_f1_and_threshold(
                scores, labels, high_score_more_similar
            )
        )
        assert f1 == pytest.approx(0.66666, TOL)
        assert precision == pytest.approx(1.0, TOL)
        assert recall == pytest.approx(0.5, TOL)
        assert f1_threshold == pytest.approx(7.95999, TOL)

    def test_ap(self):
        scores = [6.12, 5.39, 5.28, 5.94, 6.34, 6.47, 7.88, 6.62, 8.04, 5.9]
        labels = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0]
        high_score_more_similar = True
        ap = PairClassificationEvaluator.ap_score(
            scores, labels, high_score_more_similar
        )
        assert ap == pytest.approx(0.7, TOL)
