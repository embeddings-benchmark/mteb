from __future__ import annotations

import logging

import numpy as np

from mteb.MTEBResults import ScoresDict

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self):
        return self.metadata_dict["min_score"]

    @property
    def max_score(self):
        return self.metadata_dict["max_score"]

    def _evaluate_subset(self, model, data_split, **kwargs) -> ScoresDict:
        normalized_scores = list(
            map(
                lambda x: (np.array(x) - self.min_score)
                / (self.max_score - self.min_score),
                data_split["relevance"],
            )
        )
        evaluator = SummarizationEvaluator(
            machine_summaries=data_split["machine_summaries"],
            human_summaries=data_split["human_summaries"],
            texts=data_split["text"],
            gold_scores=normalized_scores,
            **kwargs,
        )
        scores = evaluator(model)
        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        m_score = self.metadata.main_score
        dist, metric = m_score.split("_")
        dist_mapping = {"cosine": "cos_sim"}
        scores["main_score"] = scores[dist_mapping.get(dist, dist)][metric]
