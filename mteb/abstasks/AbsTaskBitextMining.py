from __future__ import annotations

import logging

from ..evaluation.evaluators import BitextMiningEvaluator
from .AbsTask import AbsTask
from .MTEBResults import ScoresDict

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

    def _evaluate_subset(self, model, data_split, **kwargs) -> ScoresDict:
        if len(data_split["sentence1"]) == 1:
            sentence1 = data_split["sentence1"][0]
        else:
            sentence1 = data_split["sentence1"]
        if len(data_split["sentence2"]) == 1:
            sentence2 = data_split["sentence2"][0]
        else:
            sentence2 = data_split["sentence2"]

        if "gold" not in data_split.features:
            assert len(sentence1) == len(sentence2), "Wrong dataset format"
            n = len(sentence1)
            gold = list(zip(range(n), range(n)))
        else:
            gold = data_split["gold"]
            if len(gold) == 1:
                gold = gold[0]
            # MTEB currently only loads GOLD labels for BUCC, which is 1-indexed
            # If a 2nd 0-indexed dataset is added, it'd be cleaner to update BUCC on the Hub to be 0-indexed
            gold = [(i - 1, j - 1) for (i, j) in gold]
            assert all(
                [(i > 0) and (j > 0) for i, j in gold]
            ), "Found negative gold indices. This may be caused by MTEB expecting 1-indexed gold labels."

        evaluator = BitextMiningEvaluator(sentence1, sentence2, gold, **kwargs)
        metrics = evaluator(model)
        self._add_main_score(metrics)
        return metrics

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]
