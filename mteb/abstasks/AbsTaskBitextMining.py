import logging

from ..evaluation.evaluators import BitextMiningEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskBitextMining(AbsTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split, **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_crosslingual:
            scores = {}
            for lang in self.dataset:
                logger.info(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                data_split = self.dataset[lang][split]
                scores[lang] = self._evaluate_split(model, data_split, split_langs=lang, **kwargs)
        else:
            logger.info(f"\nTask: {self.description['name']}, split: {split}. Running...")
            data_split = self.dataset[split]
            scores = self._evaluate_split(model, data_split, split_langs=self.get_language(), **kwargs)

        return scores

    def _evaluate_split(self, model, data_split, split_langs, **kwargs):
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

        evaluator = BitextMiningEvaluator(sentence1, sentence2, gold, language=split_langs, **kwargs)
        metrics = evaluator(model)
        self._add_main_score(metrics)
        return metrics

    def _add_main_score(self, scores):
        if self.description["main_score"] in scores:
            scores["main_score"] = scores[self.description["main_score"]]
        else:
            logger.warn(f"main score {self.description['main_score']} not found in scores {scores.keys()}")
