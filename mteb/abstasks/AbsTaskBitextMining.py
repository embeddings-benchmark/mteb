from __future__ import annotations

import logging

from datasets import Dataset

from ..evaluation.evaluators import BitextMiningEvaluator
from .AbsTask import AbsTask
from .MTEBResults import HFSubset, ScoresDict

logger = logging.getLogger(__name__)


class AbsTaskBitextMining(AbsTask):
    """Abstract class for BitextMining tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        id: str
        sentence1: str
        sentence2: str
    """

    parallel_subsets = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split, **kwargs) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        hf_subsets = (
            [l for l in self.dataset]
            if self.is_multilingual or self.is_crosslingual
            else ["default"]
        )

        scores = {}
        if self.parallel_subsets:
            scores["default"] = self._evaluate_subset(
                model, self.dataset[split], parallel=True, **kwargs
            )
        else:
            for hf_subet in hf_subsets:
                logger.info(
                    f"\nTask: {self.metadata_dict['name']}, split: {split}, subset: {hf_subet}. Running..."
                )

                if hf_subet not in self.dataset and hf_subet == "default":
                    data_split = self.dataset[split]
                else:
                    data_split = self.dataset[hf_subet][split]
                scores[hf_subet] = self._evaluate_subset(
                    model, data_split, subsets=["sentence1", "sentence2"], **kwargs
                )

        return scores

    def _evaluate_subset(
        self, model, data_split: Dataset, parallel=False, **kwargs
    ) -> ScoresDict:
        evaluator = BitextMiningEvaluator(data_split, **kwargs)
        metrics = evaluator(model)
        if parallel:
            for v in metrics.values():
                self._add_main_score(v)
        else:
            self._add_main_score(metrics)
        return metrics

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]
