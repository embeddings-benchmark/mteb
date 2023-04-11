import pytest

from mteb.evaluation.evaluators import RetrievalEvaluator

TOL = 0.0001


class TestRetrievalEvaluator:
    def setup_method(self):
        """setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        queries = {str(i): "" for i in range(3)}
        corpus = {str(i): "" for i in range(10)}
        relevant_docs = {"0": set(["0", "1", "2"]), "1": set(["9"]), "2": set(["4", "5", "9"])}

        self.evaluator = RetrievalEvaluator(
            queries,
            corpus,
            relevant_docs,
            mrr_at_k=[],
            ndcg_at_k=[],
            accuracy_at_k=[],
            precision_recall_at_k=[],
            map_at_k=[],
        )

    def test_accuracy_at_k(self):
        queries_result_list = [
            [{"corpus_id": "0", "score": 0.9}, {"corpus_id": "1", "score": 0.7}],
            [{"corpus_id": "5", "score": 0.5}, {"corpus_id": "7", "score": 0.5}, {"corpus_id": "8", "score": 0.5}],
            [{"corpus_id": "4", "score": 0.9}],
        ]
        self.evaluator.accuracy_at_k = [1, 2, 3]

        assert self.evaluator._compute_metrics(queries_result_list) == {
            "accuracy_at_1": pytest.approx(0.66666, TOL),
            "accuracy_at_2": pytest.approx(0.66666, TOL),
            "accuracy_at_3": pytest.approx(0.66666, TOL),
        }

    def test_ndcg_at_k(self):
        queries_result_list = [
            [{"corpus_id": "0", "score": 0.9}, {"corpus_id": "1", "score": 0.7}],
            [{"corpus_id": "5", "score": 0.5}, {"corpus_id": "7", "score": 0.5}, {"corpus_id": "8", "score": 0.5}],
            [{"corpus_id": "4", "score": 0.9}],
        ]
        self.evaluator.ndcg_at_k = [1, 2, 3]

        assert self.evaluator._compute_metrics(queries_result_list) == {
            "ndcg_at_1": pytest.approx(0.66666, TOL),
            "ndcg_at_2": pytest.approx(0.53771, TOL),
            "ndcg_at_3": pytest.approx(0.41154, TOL),
        }

    def test_mrr_at_k(self):
        queries_result_list = [
            [{"corpus_id": "0", "score": 0.9}, {"corpus_id": "1", "score": 0.7}],
            [{"corpus_id": "5", "score": 0.5}, {"corpus_id": "7", "score": 0.5}, {"corpus_id": "8", "score": 0.5}],
            [{"corpus_id": "4", "score": 0.9}],
        ]
        self.evaluator.mrr_at_k = [1, 2, 3]

        assert self.evaluator._compute_metrics(queries_result_list) == {
            "mrr_at_1": pytest.approx(0.66666, TOL),
            "mrr_at_2": pytest.approx(0.66666, TOL),
            "mrr_at_3": pytest.approx(0.66666, TOL),
        }

    def test_map_at_k(self):
        queries_result_list = [
            [{"corpus_id": "0", "score": 0.9}, {"corpus_id": "1", "score": 0.7}],
            [{"corpus_id": "5", "score": 0.5}, {"corpus_id": "7", "score": 0.5}, {"corpus_id": "8", "score": 0.5}],
            [{"corpus_id": "4", "score": 0.9}],
        ]
        self.evaluator.map_at_k = [1, 2, 3]

        assert self.evaluator._compute_metrics(queries_result_list) == {
            "map_at_1": pytest.approx(0.66666, TOL),
            "map_at_2": pytest.approx(0.5, TOL),
            "map_at_3": pytest.approx(0.33333, TOL),
        }

    def test_precision_recall_at_k(self):
        queries_result_list = [
            [{"corpus_id": "0", "score": 0.9}, {"corpus_id": "1", "score": 0.7}],
            [{"corpus_id": "5", "score": 0.5}, {"corpus_id": "7", "score": 0.5}, {"corpus_id": "8", "score": 0.5}],
            [{"corpus_id": "4", "score": 0.9}],
        ]
        self.evaluator.precision_recall_at_k = [1, 2, 3]

        assert self.evaluator._compute_metrics(queries_result_list) == {
            "precision_at_1": pytest.approx(0.66666, TOL),
            "precision_at_2": pytest.approx(0.5, TOL),
            "precision_at_3": pytest.approx(0.33333, TOL),
            "recall_at_1": pytest.approx(0.22222, TOL),
            "recall_at_2": pytest.approx(0.33333, TOL),
            "recall_at_3": pytest.approx(0.33333, TOL),
        }
