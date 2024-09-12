from __future__ import annotations

import logging
from typing import Any

from ...evaluation.evaluators import VisualSTSEvaluator
from ...load_results.mteb_results import ScoresDict
from ..AbsTask import AbsTask, DescriptiveStatistics

logger = logging.getLogger(__name__)


class VisualSTSDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for STS

    Attributes:
        num_samples: number of samples in the dataset
        avg_score: Average score
    """

    # TODO: what are useful stats for visual STS tasks?
    # average_pixel_width; average_pixel_height; average non-white boxes?

    num_samples: int
    avg_score: float


class AbsTaskVisualSTS(AbsTask):
    """Abstract class for visual STS experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sentence1: PIL.Image
        sentence2: PIL.Image
        score: float
    """

    sentences_column_names = ["sentence1", "sentence2"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self) -> int:
        return self.metadata_dict["min_score"]

    @property
    def max_score(self) -> int:
        return self.metadata_dict["max_score"]

    def _evaluate_subset(
        self, model, data_split, *, encode_kwargs: dict[str, Any] = {}, **kwargs
    ) -> ScoresDict:
        def normalize(x):
            return (x - self.min_score) / (self.max_score - self.min_score)

        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = VisualSTSEvaluator(
            data_split,
            self.sentences_column_names,
            normalized_scores,
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)

        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> VisualSTSDescriptiveStatistics:
        if hf_subset:
            score = self.dataset[hf_subset][split]["score"]
        elif compute_overall:
            score = []
            for hf_subset in self.metadata.eval_langs:
                score.extend(self.dataset[hf_subset][split]["score"])
        else:
            score = self.dataset[split]["score"]

        avg_score = sum(score) / len(score)
        return VisualSTSDescriptiveStatistics(
            num_samples=len(score),
            avg_score=avg_score,
        )
