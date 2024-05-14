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
                scores = self._evaluate_split(model, self.dataset[split], **kwargs)
            else:
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
            print(data_split)
            scores = self._evaluate_split(
                model, data_split, subsets=["sentence1", "sentence2"], **kwargs
            )

        return scores

    def _evaluate_split(self, model, data_split, **kwargs):
        if "gold" in data_split.features:
            data_split["sentence1"] = data_split["sentence1"][0]
            data_split["sentence2"] = data_split["sentence2"][0]
            gold = data_split["gold"]
            if len(gold) == 1:
                gold = gold[0]
            # MTEB currently only loads GOLD labels for BUCC, which is 1-indexed
            # If a 2nd 0-indexed dataset is added, it'd be cleaner to update BUCC on the Hub to be 0-indexed
            gold = [(i - 1, j - 1) for (i, j) in gold]
            assert all(
                [(i > 0) and (j > 0) for i, j in gold]
            ), "Found negative gold indices. This may be caused by MTEB expecting 1-indexed gold labels."

            data_split["sentence1"] = [
                data_split["sentence1"][i] for (i, j) in self.gold
            ]

        evaluator = BitextMiningEvaluator(data_split, **kwargs)
        metrics = evaluator(model)
        self._add_main_score(metrics)
        return metrics

    def _add_main_score(self, scores):
        if self.metadata_dict["main_score"] in scores:
            scores["main_score"] = scores[self.metadata_dict["main_score"]]
        else:
            logger.warn(
                f"main score {self.metadata_dict['main_score']} not found in scores {scores.keys()}"
            )
