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

    @property
    def min_score(self):
        return self.description["min_score"]

    @property
    def max_score(self):
        return self.description["max_score"]

    def evaluate(self, model, split):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.description["available_langs"]:
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split)
        else:
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split)

        return scores

    def _evaluate_split(self, model, data_split):
        normalize = lambda x: (x - self.min_score) / (self.max_score - self.min_score)
        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(data_split["sentence1"], data_split["sentence2"], normalized_scores)
        metrics = evaluator(model)
        return metrics