from __future__ import annotations

import pytest

from mteb.evaluation.evaluators import RetrievalEvaluator

TOL = 0.0001


class TestRetrievalEvaluator:
    def setup_method(self):
        """Setup any state tied to the execution of the given method in a class.

        setup_method is invoked for every test method of a class.
        """
        self.evaluator = RetrievalEvaluator(task_name="test")

    def test_metrics_at_k(self):
        # Qid: {Docid: Relevance}
        relevant_docs = {
            "0": {"0": 1, "1": 1},
            "1": {"1": 1},
        }
        results = {
            "0": {"0": 1.0, "1": 0.9, "2": 0.8},
            "1": {"0": 0.0, "1": 1.0, "2": 0.0},
        }

        ndcg, _map, recall, precision, _ = self.evaluator.evaluate(
            relevant_docs,
            results,
            [1, 2, 3],
        )

        assert ndcg == {"NDCG@1": 0.5, "NDCG@2": 0.30657, "NDCG@3": 0.30657}
        assert _map == {"MAP@1": 0.25, "MAP@2": 0.25, "MAP@3": 0.25}
        assert recall == {"Recall@1": 0.25, "Recall@2": 0.25, "Recall@3": 0.25}
        assert precision == {"P@1": 0.5, "P@2": 0.25, "P@3": 0.16667}

    def test_nAUC(self):
        relevant_docs = {
            "0": {"0": 1, "1": 1},
            "1": {"0": 1},
            "2": {"0": 1, "1": 1, "2": 1},
            "3": {"0": 1},
            "4": {"0": 1, "1": 1},
        }
        results = {
            "0": {"0": 0.8, "1": 0.3, "2": 0.4},
            "1": {"0": 0.5, "1": 0.8, "2": 0.4},
            "2": {"0": 0.9, "1": 0.3, "2": 0.3},
            "3": {"0": 0.1, "1": 0.2, "2": 0.2},
            "4": {"0": 0.5, "1": 0.4, "2": 0.5},
        }

        _, _, _, _, naucs = self.evaluator.evaluate(
            relevant_docs,
            results,
            [1, 2, 3],
        )

        assert naucs["nAUC_NDCG@3_max"] == pytest.approx(0.62792, TOL)
        assert naucs["nAUC_NDCG@3_std"] == pytest.approx(0.06211, TOL)
        assert naucs["nAUC_NDCG@3_diff1"] == pytest.approx(0.06600, TOL)
