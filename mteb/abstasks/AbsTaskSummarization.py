from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mteb.encoder_interface import Encoder
from mteb.load_results.mteb_results import ScoresDict

from ..evaluation.evaluators import SummarizationEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskSummarization(AbsTask):
    """Abstract class for summarization experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        text: str
        human_summaries: list[str]
        machine_summaries: list[str]
        relevance: list[float] (the score of the machine generated summaries)
    """

    evalutor = SummarizationEvaluator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self):
        return self.metadata_dict["min_score"]

    @property
    def max_score(self):
        return self.metadata_dict["max_score"]

    def _evaluate_subset(
        self, model: Encoder, data_split, *, encode_kwargs: dict[str, Any], **kwargs
    ) -> ScoresDict:
        normalized_scores = [
            (np.array(x) - self.min_score) / (self.max_score - self.min_score)
            for x in data_split["relevance"]
        ]
        evaluator = self.evalutor(
            machine_summaries=data_split["machine_summaries"],
            human_summaries=data_split["human_summaries"],
            texts=data_split["text"],
            gold_scores=normalized_scores,
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]
