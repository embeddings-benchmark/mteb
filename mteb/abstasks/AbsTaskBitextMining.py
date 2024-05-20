from __future__ import annotations

import logging

from ..evaluation.evaluators import BitextMiningEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskBitextMining(AbsTask):
    """Abstract class for BitextMining tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        id: str
        sentence1: str
        sentence2: str
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        parallel_subsets = (
            self.parallel_subsets if hasattr(self, "parallel_subsets") else False
        )
        if self.is_crosslingual:
            if parallel_subsets:
                scores = self._evaluate_split(
                    model, self.dataset[split], True, **kwargs
                )
            else:
                scores = {}
                for lang in self.dataset:
                    logger.info(
                        f"\nTask: {self.metadata_dict['name']}, split: {split}, language: {lang}. Running..."
                    )
                    data_split = self.dataset[lang][split]
                    scores[lang] = self._evaluate_split(
                        model, data_split, subsets=["sentence1", "sentence2"], **kwargs
                    )
        else:
            logger.info(
                f"\nTask: {self.metadata_dict['name']}, split: {split}. Running..."
            )
            data_split = self.dataset[split]
            print(data_split)
            scores = self._evaluate_split(
                model, data_split, subsets=["sentence1", "sentence2"], **kwargs
            )

        return scores

    def _evaluate_split(self, model, data_split, parallel=False, **kwargs):
        evaluator = BitextMiningEvaluator(data_split, **kwargs)
        metrics = evaluator(model)
        if parallel:
            for v in metrics.values():
                self._add_main_score(v)
        else:
            self._add_main_score(metrics)
        return metrics

    def _add_main_score(self, scores):
        if self.metadata_dict["main_score"] in scores:
            scores["main_score"] = scores[self.metadata_dict["main_score"]]
        else:
            logger.warn(
                f"main score {self.metadata_dict['main_score']} not found in scores {scores.keys()}"
            )
