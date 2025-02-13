from __future__ import annotations

from mteb.evaluation.evaluators import RetrievalEvaluator, utils
from mteb.models import SentenceTransformerWrapper
from tests.test_benchmark.mock_models import MockNumpyEncoder


class TestInstructionMetricsEvaluation:
    def setup_method(self):
        """Setup any state tied to the execution of the given method in a class.

        setup_method is invoked for every test method of a class.
        """
        # checks that it loads
        self.evaluator = RetrievalEvaluator(
            SentenceTransformerWrapper(MockNumpyEncoder()), task_name="test"
        )

    def test_p_mrr(self):
        changed_qrels = {
            "a": ["0"],
        }

        # these are the query: {"doc_id": score}
        original_run = {
            "a-og": {"0": 1, "1": 2, "2": 3, "3": 4},
        }

        new_run = {
            "a-changed": {"0": 1, "1": 2, "2": 3, "3": 4},
        }

        score = utils.calculate_pmrr(
            original_run,
            new_run,
            changed_qrels,
        )
        assert score == 0.0

        # test with a change

        new_run = {
            "a-changed": {"0": 4, "1": 1, "2": 2, "3": 3},
        }

        score = utils.calculate_pmrr(
            original_run,
            new_run,
            changed_qrels,
        )
        assert score == -0.75

        # test with a positive change, flipping them
        new_run = {
            "a-og": {"0": 4, "1": 1, "2": 2, "3": 3},
        }
        original_run = {
            "a-changed": {"0": 1, "1": 2, "2": 3, "3": 4},
        }
        score = utils.calculate_pmrr(
            new_run,
            original_run,
            changed_qrels,
        )
        assert score == 0.75
