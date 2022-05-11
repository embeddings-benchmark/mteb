from .AbsTask import AbsTask
from ..evaluation.evaluators import STSEvaluator
import datasets
import numpy as np
import logging


class AbsTaskSTS(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """

    def __init__(self, **kwargs):
        super(AbsTaskSTS, self).__init__(**kwargs)
        self.dataset = None
        self.data_loaded = False

    @property
    def min_score(self):
        return self.description["min_score"]

    @property
    def max_score(self):
        return self.description["max_score"]

    def load_data(self):
        if self.data_loaded:
            return

        self.dataset = datasets.load_dataset(self.description["hf_hub_name"])
        self.data_loaded = True

    def evaluate(self, model, split):
        if not self.data_loaded:
            self.load_data()

        data_split = self.dataset[split]
        normalize = lambda x: (x - self.min_score) / (self.max_score - self.min_score)
        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(
            data_split["sentence1"], data_split["sentence2"], normalized_scores
        )
        metrics = evaluator(model)

        return metrics
