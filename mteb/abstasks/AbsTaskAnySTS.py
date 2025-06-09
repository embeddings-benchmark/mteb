from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from mteb.abstasks.TaskMetadata import (
    DescriptiveStatistics,
    ImageStatistics,
    ScoreStatistics,
    TextStatistics,
)
from mteb.encoder_interface import Encoder
from mteb.types import ScoresDict

from ..evaluation.evaluators import AnySTSEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AnySTSDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for STS

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of unique pairs

        text1_statistics: Statistics for sentence1
        text2_statistics: Statistics for sentence2

        image1_statistics: Statistics for image1
        image2_statistics: Statistics for image2

        label_statistics: Statistics for labels
    """

    num_samples: int
    number_of_characters: int | None
    unique_pairs: int | None

    text1_statistics: TextStatistics | None
    text2_statistics: TextStatistics | None

    image1_statistics: ImageStatistics | None
    image2_statistics: ImageStatistics | None

    label_statistics: ScoreStatistics


class AbsTaskAnySTS(AbsTask):
    """Abstract class for STS experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns::
        sentence1: str
        sentence2: str
        score: float
    """

    abstask_prompt = "Retrieve semantically similar text."
    column_names: tuple[str, str] = ("sentence1", "sentence2")
    min_score: int = 0
    max_score: int = 5

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        **kwargs: Any,
    ) -> ScoresDict:
        normalized_scores = list(map(self.normalize, data_split["score"]))
        evaluator = AnySTSEvaluator(
            data_split,
            self.column_names,
            normalized_scores,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        return scores

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> AnySTSDescriptiveStatistics:
        first_column, second_column = self.column_names
        if hf_subset:
            sentence1 = self.dataset[hf_subset][split][first_column]
            sentence2 = self.dataset[hf_subset][split][second_column]
            score = self.dataset[hf_subset][split]["score"]
        elif compute_overall:
            sentence1 = []
            sentence2 = []
            score = []
            for hf_subset in self.metadata.eval_langs:
                sentence1.extend(self.dataset[hf_subset][split][first_column])
                sentence2.extend(self.dataset[hf_subset][split][second_column])
                score.extend(self.dataset[hf_subset][split]["score"])
        else:
            sentence1 = self.dataset[split][first_column]
            sentence2 = self.dataset[split][second_column]
            score = self.dataset[split]["score"]

        if "text" in self.metadata.modalities:
            text1_statistics = TextStatistics(
                min_text_length=min(len(s) for s in sentence1),
                average_text_length=sum(len(s) for s in sentence1) / len(sentence1),
                max_text_length=max(len(s) for s in sentence1),
                unique_texts=len(set(sentence1)),
            )
            text2_statistics = TextStatistics(
                min_text_length=min(len(s) for s in sentence2),
                max_text_length=max(len(s) for s in sentence2),
                average_text_length=sum(len(s) for s in sentence2) / len(sentence2),
                unique_texts=len(set(sentence2)),
            )
            sentence1_len = [len(s) for s in sentence1]
            sentence2_len = [len(s) for s in sentence2]
            number_of_characters = sum(sentence1_len) + sum(sentence2_len)
            unique_pairs = len(set(zip(sentence1, sentence2)))
        else:
            text1_statistics = None
            text2_statistics = None
            number_of_characters = None
            unique_pairs = None

        if "image" in self.metadata.modalities:
            img_widths1, img_heights1 = [], []
            for img in sentence1:
                width, height = img.size
                img_heights1.append(height)
                img_widths1.append(width)

            image1_statistics = ImageStatistics(
                min_image_width=min(img_widths1),
                average_image_width=sum(img_widths1) / len(img_widths1),
                max_image_width=max(img_widths1),
                min_image_height=min(img_heights1),
                average_image_height=sum(img_heights1) / len(img_heights1),
                max_image_height=max(img_widths1),
            )

            img_widths2, img_heights2 = [], []
            for img in sentence2:
                width, height = img.size
                img_heights2.append(height)
                img_widths2.append(width)

            image2_statistics = ImageStatistics(
                min_image_width=min(img_widths2),
                average_image_width=sum(img_widths2) / len(img_widths2),
                max_image_width=max(img_widths2),
                min_image_height=min(img_heights2),
                average_image_height=sum(img_heights2) / len(img_heights2),
                max_image_height=max(img_widths2),
            )
        else:
            image1_statistics = None
            image2_statistics = None

        labels_statistics = ScoreStatistics(
            min_score=min(score),
            avg_score=sum(score) / len(score),
            max_score=max(score),
        )

        return AnySTSDescriptiveStatistics(
            num_samples=len(sentence1),
            number_of_characters=number_of_characters,
            unique_pairs=unique_pairs,
            text1_statistics=text1_statistics,
            text2_statistics=text2_statistics,
            image1_statistics=image1_statistics,
            image2_statistics=image2_statistics,
            label_statistics=labels_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self._upload_dataset_to_hub(
            repo_name, [self.column_names[0], self.column_names[1], "score"]
        )

    def normalize(self, x: float) -> float:
        return (x - self.min_score) / (self.max_score - self.min_score)
