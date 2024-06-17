from __future__ import annotations

import pytest

from mteb.evaluation.evaluators import RetrievalEvaluator

TOL = 0.0001


class TestRetrievalEvaluator:
    def setup_method(self):
        """Setup any state tied to the execution of the given method in a class.

        setup_method is invoked for every test method of a class.
        """
        self.evaluator = RetrievalEvaluator()

    @pytest.mark.parametrize(
        "relevant_docs, results, ignore_identical_ids, expected_metrics",
        [
            (
                # Qid: {Docid: Relevance}
                {
                    "0": {"0": 1, "1": 1},
                    "1": {"1": 1},
                },
                {
                    "0": {"0": 1.0, "1": 0.9, "2": 0.8},
                    "1": {"0": 0.0, "1": 1.0, "2": 0.0},
                },
                False, 
                {
                "ndcg": {'NDCG@1': 1.0, 'NDCG@2': 1.0, 'NDCG@3': 1.0},
                "map": {"MAP@1": 0.75, "MAP@2": 1.0, "MAP@3": 1.0},
                "recall": {"Recall@1": 0.75, "Recall@2": 1.0, "Recall@3": 1.0},
                "precision": {"P@1": 1.0, "P@2": 0.75, "P@3": 0.5},
                }
            ),
            (
                # Qid: {Docid: Relevance}
                {
                    "0": {"0": 1, "1": 1},
                    "1": {"1": 1},
                },
                {
                    "0": {"0": 1.0, "1": 0.9, "2": 0.8},
                    "1": {"0": 0.0, "1": 1.0, "2": 0.0},
                },
                True, 
                {
                "ndcg": {'NDCG@1': 1.0, 'NDCG@2': 1.0, 'NDCG@3': 1.0, 'NDCG(no self)@1': 0.5, 'NDCG(no self)@2': 0.30657, 'NDCG(no self)@3': 0.30657},
                 "map": {"MAP@1": 0.25, "MAP@2": 0.25, "MAP@3": 0.25},
                "recall": {"Recall@1": 0.25, "Recall@2": 0.25, "Recall@3": 0.25},
                "precision": {"P@1": 0.5, "P@2": 0.25, "P@3": 0.16667},
                }
            ),
        ],
    )
    def test_metrics_at_k(self, relevant_docs, results, ignore_identical_ids, expected_metrics):
        ndcg, _map, recall, precision = self.evaluator.evaluate(
            relevant_docs,
            results,
            [1, 2, 3],
            ignore_identical_ids=ignore_identical_ids,
        )

        assert ndcg == expected_metrics["ndcg"]
        assert _map == expected_metrics["map"]
        assert recall == expected_metrics["recall"]
        assert precision == expected_metrics["precision"]
