import pytest

from mteb.evaluation.evaluators import RetrievalEvaluator

TOL = 0.0001

class TestRetrievalEvaluator:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """

        self.evaluator = RetrievalEvaluator()

    def test_ndcg_at_k(self):
        # Qid: {Docid: Relevance}
        relevant_docs = {
            "0": {"0": 1, "1": 1},
            "1": {"1": 1},
        }
        results = {
            "0": {"0": 1.0, "1": 0.9, "2": 0.8},
            "1": {"0": 0.0, "1": 1.0, "2": 0.0},
        }

        ndcg, _map, recall, precision = self.evaluator.evaluate(
            relevant_docs,
            results,
            [1, 2, 3],
        )

        assert ndcg == {'NDCG@1': 0.5, 'NDCG@2': 0.30657, 'NDCG@3': 0.30657}
        assert _map == {'MAP@1': 0.25, 'MAP@2': 0.25, 'MAP@3': 0.25}
        assert recall == {'Recall@1': 0.25, 'Recall@2': 0.25, 'Recall@3': 0.25}
        assert precision == {'P@1': 0.5, 'P@2': 0.25, 'P@3': 0.16667}
