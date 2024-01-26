import logging

import numpy as np

from ..evaluation.evaluators import SummarizationEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskSummarization(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self):
        return self.description["min_score"]

    @property
    def max_score(self):
        return self.description["max_score"]

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                logger.info(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, language=lang, **kwargs)
        else:
            logger.info(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, language=self.get_language(), **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, language, **kwargs):
        normalized_scores = list(
            map(lambda x: (np.array(x) - self.min_score) / (self.max_score - self.min_score), data_split["relevance"])
        )
        evaluator = SummarizationEvaluator(
            machine_summaries=data_split["machine_summaries"],
            human_summaries=data_split["human_summaries"],
            texts=data_split["text"],
            gold_scores=normalized_scores,
            language=language,
            **kwargs,
        )
        metrics = evaluator(model)
        return metrics
