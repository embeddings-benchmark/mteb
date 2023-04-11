import pytest

from mteb.evaluation.evaluators import RerankingEvaluator

TOL = 0.0001


class TestRerankingEvaluator:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """

        self.evaluator = RerankingEvaluator([])

    def test_mrr_at_k(self):
        is_relevant = [1, 1, 1, 0, 0, 0, 0, 0, 0]
        pred_ranking = [5, 2, 6, 1, 3, 4, 7, 8, 9]

        assert self.evaluator.mrr_at_k_score(is_relevant, pred_ranking, 10) == pytest.approx(0.5, TOL)
        assert self.evaluator.mrr_at_k_score(is_relevant, pred_ranking, 3) == pytest.approx(0.5, TOL)
        assert self.evaluator.mrr_at_k_score(is_relevant, pred_ranking, 1) == pytest.approx(0, TOL)

    def test_map(self):
        is_relevant = [1, 1, 1, 0, 0]
        pred_scores = [0.75, 0.93, 0.85, 0.76, 0.75]

        assert self.evaluator.ap_score(is_relevant, pred_scores) == pytest.approx(0.86666, TOL)
