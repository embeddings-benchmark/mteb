import logging

from ..evaluation.evaluators import STSEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskSTS(AbsTask):
    """
    Abstract class for STS experiments.

    self.load_data() must return a huggingface dataset containing a test split, and the following columns:
        sentence1: str
        sentence2: str
        score: float
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

        if self.is_crosslingual or self.is_multilingual:
            scores = {}
            for lang in self.dataset:
                logger.info(
                    f"Task: {self.description['name']}, split: {split}, language: {lang}. Running..."
                )
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            logger.info(
                f"\nTask: {self.description['name']}, split: {split}. Running..."
            )
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, **kwargs):
        def normalize(x):
            return (x - self.min_score) / (self.max_score - self.min_score)

        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(
            data_split["sentence1"],
            data_split["sentence2"],
            normalized_scores,
            **kwargs,
        )
        metrics = evaluator(model)
        return metrics
