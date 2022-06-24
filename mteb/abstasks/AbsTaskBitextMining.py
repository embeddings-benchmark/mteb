import random

import datasets
import numpy as np
import tqdm

from ..evaluation.evaluators import BitextMiningEvaluator
from .AbsTask import AbsTask


class AbsTaskBitextMining(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = 42

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, **kwargs)
        else:
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, **kwargs):
        if len(data_split["sentence1"]) == 1:
            sentence1 = data_split["sentence1"][0]
        else:
            sentence1 = data_split["sentence1"]
        if len(data_split["sentence2"]) == 1:
            sentence2 = data_split["sentence2"][0]
        else:
            sentence2 = data_split["sentence2"]

        if not ("gold" in data_split.features):
            assert len(sentence1) == len(sentence2), "Wrong dataset format"
            n = len(sentence1)
            gold = list(zip(range(n), range(n)))
        else:
            gold = data_split["gold"]
        if len(gold) == 1:
            gold = gold[0]
        if max([i for (i, j) in gold]) == len(sentence1):
            gold = [(i - 1, j - 1) for (i, j) in gold]

        evaluator = BitextMiningEvaluator(sentence1, sentence2, gold, **kwargs)
        metrics = evaluator(model)
        return metrics
