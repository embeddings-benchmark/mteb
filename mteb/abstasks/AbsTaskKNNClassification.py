from .AbsTask import AbsTask
import datasets
from ..evaluation.evaluators import kNNClassificationEvaluator
import numpy as np
import logging
from collections import defaultdict


class AbsTaskKNNClassification(AbsTask):
    """
    Abstract class for kNN classification tasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for classification. #TODO:
    """

    def __init__(self, **kwargs):
        super(AbsTaskKNNClassification, self).__init__(**kwargs)
        self.dataset = None
        self.data_loaded = False

    def load_data(self):
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(self.description["hf_hub_name"])
        self.data_loaded = True

    def evaluate(self, model, eval_split="test", train_split="train"):
        if not self.data_loaded:
            self.load_data()

        train_split = self.dataset[train_split]
        eval_split = self.dataset[eval_split]

        logging.getLogger("sentence_transformers.evaluation.kNNClassificationEvaluator").setLevel(logging.WARN)
        evaluator = kNNClassificationEvaluator(
            train_split["text"], train_split["label"], eval_split["text"], eval_split["label"]
        )
        scores = evaluator(model)
        return scores
