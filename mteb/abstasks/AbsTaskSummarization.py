import logging

import numpy as np

from ..evaluation.evaluators import SummarizationEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskSummarization(AbsTask):
    """
    Abstract class for summarization experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        text: str
        human_summaries: list[str]
        machine_summaries: list[str]
        relevance: list[float] (the score of the machine generated summaries)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self):
        return self.metadata_dict["min_score"]

    @property
    def max_score(self):
        return self.metadata_dict["max_score"]

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                logger.info(
                    f"\nTask: {self.metadata_dict['name']}, split: {split}, language: {lang}. Running..."
                )
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            logger.info(
                f"\nTask: {self.metadata_dict['name']}, split: {split}. Running..."
            )
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, **kwargs):
        normalized_scores = list(
            map(
                lambda x: (np.array(x) - self.min_score)
                / (self.max_score - self.min_score),
                data_split["relevance"],
            )
        )
        evaluator = SummarizationEvaluator(
            machine_summaries=data_split["machine_summaries"],
            human_summaries=data_split["human_summaries"],
            texts=data_split["text"],
            gold_scores=normalized_scores,
            **kwargs,
        )
        metrics = evaluator(model)
        return metrics
