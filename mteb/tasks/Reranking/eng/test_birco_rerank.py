from __future__ import annotations

import unittest

from sentence_transformers import SentenceTransformer

from mteb import MTEB

from . import (
    BIRCOArguAnaReranking,
    BIRCOClinicalTrialReranking,
    BIRCODorisMaeReranking,
    BIRCORelicReranking,
    BIRCOWhatsThatBookReranking,
)


class TestBIRCOTasks(unittest.TestCase):
    def setUp(self):
        # Initialize a reference model for testing
        self.model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        # Instantiate the BIRCO tasks
        self.tasks = [
            BIRCODorisMaeReranking(),
            BIRCOArguAnaReranking(),
            BIRCOClinicalTrialReranking(),
            BIRCOWhatsThatBookReranking(),
            BIRCORelicReranking(),
        ]

    def test_run_tasks(self):
        evaluation = MTEB(tasks=self.tasks)
        results = evaluation.run(self.model)
        # Check that each task produces non-empty results and print them for manual inspection.
        for task_name, task_results in results.items():
            self.assertTrue(task_results, f"No results returned for task: {task_name}")
            print(f"Results for {task_name}: {task_results}")


if __name__ == "__main__":
    # Define the sentence-transformers model name
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    model = SentenceTransformer(model_name)
    tasks = [
        BIRCODorisMaeReranking(),
        BIRCOArguAnaReranking(),
        BIRCOClinicalTrialReranking(),
        BIRCOWhatsThatBookReranking(),
        BIRCORelicReranking(),
    ]
    evaluation = MTEB(tasks=tasks)
