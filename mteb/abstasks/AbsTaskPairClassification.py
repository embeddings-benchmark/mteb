from __future__ import annotations

import logging
from collections import defaultdict

from ..evaluation.evaluators import PairClassificationEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskPairClassification(AbsTask):
    """
    Abstract class for PairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sent1: list[str]
        sent2: list[str]
        labels: list[int]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate_monolingual(self, model, dataset, split="test", **kwargs):
        data_split = dataset[split][0]
        logging.getLogger(
            "sentence_transformers.evaluation.PairClassificationEvaluator"
        ).setLevel(logging.WARN)
        evaluator = PairClassificationEvaluator(
            data_split["sent1"], data_split["sent2"], data_split["labels"], **kwargs
        )
        scores = evaluator.compute_metrics(model)

        # Compute max
        max_scores = defaultdict(list)
        for sim_fct in scores:
            for metric in ["accuracy", "f1", "ap"]:
                max_scores[metric].append(scores[sim_fct][metric])

        for metric in max_scores:
            max_scores[metric] = max(max_scores[metric])

        scores["max"] = dict(max_scores)

        return scores

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()
        if self.is_multilingual:
            scores = dict()
            print("loaded langs:", self.dataset.keys())
            for lang, monolingual_dataset in self.dataset.items():
                logger.info(
                    f"\nTask: {self.metadata_dict['name']}, split: {split}, language: {lang}. Running..."
                )
                scores[lang] = self._evaluate_monolingual(
                    model, monolingual_dataset, split=split, **kwargs
                )
            return scores
        else:
            logger.info(
                f"\nTask: {self.metadata_dict['name']}, split: {split}. Running..."
            )
            return self._evaluate_monolingual(
                model, self.dataset, split=split, **kwargs
            )
