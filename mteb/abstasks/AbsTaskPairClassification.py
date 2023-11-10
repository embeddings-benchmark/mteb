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
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_multilingual:
            scores = {}
            for lang in self.dataset:
                logger.info(f"\nTask: {self.description['name']}, split: {split}, language: {lang}. Running...")
                scores[lang] = self._evaluate_monolingual(model, self.dataset[lang], split, **kwargs)
        else:
            logger.info(f"\nTask: {self.description['name']}, split: {split}. Running...")
            scores = self._evaluate_monolingual(model, self.dataset, split, **kwargs)

        return scores

    def _evaluate_monolingual(self, model, dataset, split, **kwargs):
        data_split = dataset[split][0]
        scores = []
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
