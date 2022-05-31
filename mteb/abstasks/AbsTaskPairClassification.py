from .AbsTask import AbsTask
import datasets
from sentence_transformers import evaluation
import numpy as np
import logging
from collections import defaultdict


class AbsTaskPairClassification(AbsTask):
    """
    Abstract class for PairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split="test"):
        if not self.data_loaded:
            self.load_data()

        data_split = self.dataset[split][0]

        logging.getLogger("sentence_transformers.evaluation.PairClassificationEvaluator").setLevel(logging.WARN)
        evaluator = evaluation.PairClassificationEvaluator(
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
