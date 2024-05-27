from __future__ import annotations

import logging
from collections import defaultdict

from datasets import Dataset

from ..encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from ..evaluation.evaluators import PairClassificationEvaluator
from ..MTEBResults import ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskPairClassification(AbsTask):
    """Abstract class for PairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sent1: list[str]
        sent2: list[str]
        labels: list[int]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores["max"][self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: Encoder | EncoderWithQueryCorpusEncode,
        dataset: Dataset,
        **kwargs,
    ) -> ScoresDict:
        data_split = dataset[0]
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

        self._add_main_score(scores)
        return scores
