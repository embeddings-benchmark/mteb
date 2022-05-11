from .AbsTask import AbsTask
import datasets
from sentence_transformers import evaluation
import numpy as np
import logging
from collections import defaultdict


class AbsTaskBinaryClassification(AbsTask):
    """
    Abstract class for BinaryClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise binary classification.
    """

    def __init__(self, **kwargs):
        super(AbsTaskBinaryClassification, self).__init__(**kwargs)
        self.dataset = None
        self.data_loaded = False

    def load_data(self):
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(self.description["hf_hub_name"])
        self.data_loaded = True

    def evaluate(self, model, split="test"):
        if not self.data_loaded:
            self.load_data()

        data_split = self.dataset[split][0]

        logging.getLogger("sentence_transformers.evaluation.BinaryClassificationEvaluator").setLevel(logging.WARN)
        evaluator = evaluation.BinaryClassificationEvaluator(
            data_split["sent1"], data_split["sent2"], data_split["labels"]
        )
        scores = evaluator.compute_metrices(model)

        # Compute max
        max_scores = defaultdict(list)
        for sim_fct in scores:
            for metric in ["accuracy", "f1", "ap"]:
                max_scores[metric].append(scores[sim_fct][metric])

        for metric in max_scores:
            max_scores[metric] = max(max_scores[metric])

        scores["max"] = dict(max_scores)

        return scores
