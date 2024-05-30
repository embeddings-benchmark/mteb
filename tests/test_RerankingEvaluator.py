from __future__ import annotations

import pytest

from mteb.evaluation.evaluators import RerankingEvaluator

TOL = 0.0001


class TestRerankingEvaluator:
    def setup_method(self):
        """Setup any state tied to the execution of the given method in a class.

        setup_method is invoked for every test method of a class.
        """
        self.evaluator = RerankingEvaluator([])

    def test_mrr_at_k(self):
        is_relevant = [1, 1, 1, 0, 0, 0, 0, 0, 0]
        pred_ranking = [5, 2, 6, 1, 3, 4, 7, 8, 9]

        assert self.evaluator.mrr_at_k_score(
            is_relevant, pred_ranking, 10
        ) == pytest.approx(0.5, TOL)
        assert self.evaluator.mrr_at_k_score(
            is_relevant, pred_ranking, 3
        ) == pytest.approx(0.5, TOL)
        assert self.evaluator.mrr_at_k_score(
            is_relevant, pred_ranking, 1
        ) == pytest.approx(0, TOL)

    def test_map(self):
        is_relevant = [1, 1, 1, 0, 0]
        pred_scores = [0.75, 0.93, 0.85, 0.76, 0.75]

        assert self.evaluator.ap_score(is_relevant, pred_scores) == pytest.approx(
            0.86666, TOL
        )

    def test_nAUC(self):
        is_relevant = [[1, 1, 0, 0, 0], [1, 0, 0], [1, 1, 1, 0], [1, 0], [1, 1, 0, 0]]
        pred_scores = [
            [0.8, 0.3, 0.4, 0.6, 0.5],
            [0.5, 0.8, 0.4],
            [0.9, 0.3, 0.3, 0.1],
            [0.1, 0.2],
            [0.5, 0.4, 0.5, 0.2],
        ]

        ap_scores = [self.evaluator.ap_score(y, x) for x, y in zip(pred_scores, is_relevant)]
        conf_scores = [self.evaluator.conf_scores(x) for x in pred_scores]
        nauc_scores_map = self.evaluator.nAUC_scores(conf_scores, ap_scores, "map")

        assert nauc_scores_map["nAUC_map_max"] == pytest.approx(0.69555, TOL)
        assert nauc_scores_map["nAUC_map_std"] == pytest.approx(0.86172, TOL)
        assert nauc_scores_map["nAUC_map_diff1"] == pytest.approx(0.68961, TOL)
