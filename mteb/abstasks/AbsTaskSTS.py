from __future__ import annotations

import logging

from ..evaluation.evaluators import STSEvaluator
from ..MTEBResults import ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskSTS(AbsTask):
    """Abstract class for STS experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns::
        sentence1: str
        sentence2: str
        score: float
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self) -> int:
        return self.metadata_dict["min_score"]

    @property
    def max_score(self) -> int:
        return self.metadata_dict["max_score"]

    def _evaluate_subset(self, model, data_split, **kwargs) -> ScoresDict:
        def normalize(x):
            return (x - self.min_score) / (self.max_score - self.min_score)

        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(
            data_split["sentence1"],
            data_split["sentence2"],
            normalized_scores,
            task_name=self.metadata.name,
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
