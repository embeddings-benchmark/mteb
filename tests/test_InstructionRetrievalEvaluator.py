from __future__ import annotations

from mteb.evaluation.evaluators import InstructionRetrievalEvaluator, utils


class TestInstructionRetrievalEvaluator:
    def setup_method(self):
        """Setup any state tied to the execution of the given method in a class.

        setup_method is invoked for every test method of a class.
        """
        # checks that it loads
        self.evaluator = InstructionRetrievalEvaluator.InstructionRetrievalEvaluator(task_name="test")

    def test_p_mrr(self):
        changed_qrels = {
            "a": ["0"],
        }

        # these are the query: {"doc_id": score}
        original_run = {
            "a": {"0": 1, "1": 2, "2": 3, "3": 4},
        }

        new_run = {
            "a": {"0": 1, "1": 2, "2": 3, "3": 4},
        }

        results = utils.evaluate_change(
            original_run,
            new_run,
            changed_qrels,
        )

        assert results["p-MRR"] == 0.0

        # test with a change

        new_run = {
            "a": {"0": 4, "1": 1, "2": 2, "3": 3},
        }

        results = utils.evaluate_change(
            original_run,
            new_run,
            changed_qrels,
        )

        assert results["p-MRR"] == -0.75

        # test with a positive change

        results = utils.evaluate_change(
            new_run,
            original_run,
            changed_qrels,
        )

        assert results["p-MRR"] == 0.75
