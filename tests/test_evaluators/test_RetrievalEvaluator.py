from __future__ import annotations

import pytest

from mteb import SentenceTransformerWrapper
from mteb.evaluation.evaluators import RetrievalEvaluator
from tests.test_benchmark.mock_models import MockNumpyEncoder

TOL = 0.0001


class TestRetrievalEvaluator:
    def setup_method(self):
        """Setup any state tied to the execution of the given method in a class.

        setup_method is invoked for every test method of a class.
        """
        self.evaluator = RetrievalEvaluator(
            SentenceTransformerWrapper(MockNumpyEncoder()),
        )

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
                    "ndcg": {"NDCG@1": 1.0, "NDCG@2": 1.0, "NDCG@3": 1.0},
                    "map": {"MAP@1": 0.75, "MAP@2": 1.0, "MAP@3": 1.0},
                    "recall": {"Recall@1": 0.75, "Recall@2": 1.0, "Recall@3": 1.0},
                    "precision": {"P@1": 1.0, "P@2": 0.75, "P@3": 0.5},
                },
            ),
            # Test no self retrieval
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
                    "ndcg": {"NDCG@1": 0.5, "NDCG@2": 0.30657, "NDCG@3": 0.30657},
                    "map": {"MAP@1": 0.25, "MAP@2": 0.25, "MAP@3": 0.25},
                    "recall": {"Recall@1": 0.25, "Recall@2": 0.25, "Recall@3": 0.25},
                    "precision": {"P@1": 0.5, "P@2": 0.25, "P@3": 0.16667},
                },
            ),
        ],
    )
    def test_metrics_at_k(
        self, relevant_docs, results, ignore_identical_ids, expected_metrics
    ):
        output = self.evaluator.evaluate(
            relevant_docs,
            results,
            [1, 2, 3],
            ignore_identical_ids=ignore_identical_ids,
        )

        ndcg, _map, recall, precision, nauc = output

        assert ndcg == expected_metrics["ndcg"]
        assert _map == expected_metrics["map"]
        assert recall == expected_metrics["recall"]
        assert precision == expected_metrics["precision"]

    @pytest.mark.parametrize(
        "ignore_identical_ids, expected_naucs",
        [
            (
                True,
                {
                    "nAUC_NDCG@3_max": 0.50843,
                    "nAUC_NDCG@3_std": 0.18322,
                    "nAUC_NDCG@3_diff1": 0.21416,
                },
            ),
            (
                False,
                {
                    "nAUC_NDCG@3_max": 0.8368244286523474,
                    "nAUC_NDCG@3_std": 0.9125701917627439,
                    "nAUC_NDCG@3_diff1": 0.950708977119359,
                },
            ),
        ],
    )
    def test_nAUC(self, ignore_identical_ids, expected_naucs):
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
            ignore_identical_ids=ignore_identical_ids,
        )

        print(
            naucs["nAUC_NDCG@3_max"],
            naucs["nAUC_NDCG@3_std"],
            naucs["nAUC_NDCG@3_diff1"],
        )

        aucs = ["nAUC_NDCG@3_max", "nAUC_NDCG@3_std", "nAUC_NDCG@3_diff1"]
        for auc in aucs:
            assert naucs[auc] == pytest.approx(expected_naucs[auc], TOL)
