from __future__ import annotations

import logging
from typing import Any

from ...evaluation.evaluators import VisualSTSEvaluator
from ..AbsTask import AbsTask, DescriptiveStatistics, ScoresDict

logger = logging.getLogger(__name__)


class VisualSTSDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for STS

    Attributes:
        num_samples: number of samples in the dataset

        min_image1_width: Minimum width of images1
        average_image1_width: Average width of images1
        max_image1_width: Maximum width of images1

        min_image1_height: Minimum height of images1
        average_image1_height: Average height of images1
        max_image1_height: Maximum height of images1

        min_image2_width: Minimum width of images2
        average_image2_width: Average width of images2
        max_image2_width: Maximum width of images2

        min_image2_height: Minimum height of images2
        average_image2_height: Average height of images2
        max_image2_height: Maximum height of images2

        min_score: Minimum score
        avg_score: Average score
        max_score: Maximum score
    """

    num_samples: int

    min_image1_width: float
    average_image1_width: float
    max_image1_width: float

    min_image1_height: float
    average_image1_height: float
    max_image1_height: float

    min_image2_width: float
    average_image2_width: float
    max_image2_width: float

    min_image2_height: float
    average_image2_height: float
    max_image2_height: float

    min_score: float
    avg_score: float
    max_score: float


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
            images1 = self.dataset[hf_subset][split][self.sentences_column_names[0]]
            images2 = self.dataset[hf_subset][split][self.sentences_column_names[1]]
            score = self.dataset[hf_subset][split]["score"]
        elif compute_overall:
            images1, images2 = [], []
            score = []
            for hf_subset in self.metadata.eval_langs:
                images1.extend(
                    self.dataset[hf_subset][split][self.sentences_column_names[0]]
                )
                images2.extend(
                    self.dataset[hf_subset][split][self.sentences_column_names[1]]
                )
                score.extend(self.dataset[hf_subset][split]["score"])
        else:
            images1 = self.dataset[split][self.sentences_column_names[0]]
            images2 = self.dataset[split][self.sentences_column_names[1]]
            score = self.dataset[split]["score"]

        img_widths1, img_heights1 = [], []
        for img in images1:
            width, height = img.size
            img_heights1.append(height)
            img_widths1.append(width)

        img_widths2, img_heights2 = [], []
        for img in images1:
            width, height = img.size
            img_heights2.append(height)
            img_widths2.append(width)

        return VisualSTSDescriptiveStatistics(
            num_samples=len(score),
            min_image1_width=min(img_widths1),
            average_image1_width=sum(img_widths1) / len(img_widths1),
            max_image1_width=max(img_widths1),
            min_image1_height=min(img_heights1),
            average_image1_height=sum(img_heights1) / len(img_heights1),
            max_image1_height=max(img_widths1),
            min_image2_width=min(img_widths2),
            average_image2_width=sum(img_widths2) / len(img_widths2),
            max_image2_width=max(img_widths2),
            min_image2_height=min(img_heights2),
            average_image2_height=sum(img_heights2) / len(img_heights2),
            max_image2_height=max(img_widths2),
            min_score=min(score),
            avg_score=sum(score) / len(score),
            max_score=max(score),
        )
