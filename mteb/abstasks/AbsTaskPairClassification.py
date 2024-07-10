from __future__ import annotations

import logging

from datasets import Dataset

from ..encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from ..evaluation.evaluators import PairClassificationEvaluator
from ..load_results.mteb_results import ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskPairClassification(AbsTask):
    """Abstract class for PairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sentence1: list[str]
        sentence2: list[str]
        labels: list[int]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: Encoder | EncoderWithQueryCorpusEncode,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, str] = {},
        **kwargs,
    ) -> ScoresDict:
        data_split = dataset[0]
        logging.getLogger(
            "sentence_transformers.evaluation.PairClassificationEvaluator"
        ).setLevel(logging.WARN)
        evaluator = PairClassificationEvaluator(
            data_split["sentence1"],
            data_split["sentence2"],
            data_split["labels"],
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator.compute_metrics(model, encode_kwargs=encode_kwargs)

        self._add_main_score(scores)
        return scores
