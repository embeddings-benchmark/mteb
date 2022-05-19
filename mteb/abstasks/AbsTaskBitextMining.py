from .AbsTask import AbsTask
from ..evaluation.evaluators import BitextMiningEvaluator
import datasets
import numpy as np
import tqdm
import random
import numpy as np


class AbsTaskBitextMining(AbsTask):
    def __init__(self):
        super(AbsTaskBitextMining, self).__init__()
        self.seed = 42

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
        evaluator = BitextMiningEvaluator(data_split["sentence1"], data_split["sentence2"])
        metrics = evaluator(model)
        return metrics