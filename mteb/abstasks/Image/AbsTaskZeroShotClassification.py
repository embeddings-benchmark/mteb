from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from datasets import Dataset

from ...encoder_interface import Encoder
from ...evaluation.evaluators import ZeroShotClassificationEvaluator
from ..AbsTask import AbsTask, ScoresDict
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class ZeroShotClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for ZeroShotClassification

    Attributes:
        num_samples: number of samples in the dataset.

        min_image_width: Minimum width of images
        average_image_width: Average width of images
        max_image_width: Maximum width of images

        min_image_height: Minimum height of images
        average_image_height: Average height of images
        max_image_height: Maximum height of images

        unique_labels: Number of unique labels
        labels: dict of label frequencies

        min_label_text_length: Minimum length of candidate label text
        average_label_text_length: Average length of candidate label text
        max_label_text_length: Maximum length of candidate label text
    """

    num_samples: int

    min_image_width: float
    average_image_width: float
    max_image_width: float

    min_image_height: float
    average_image_height: float
    max_image_height: float

    unique_num_labels: int
    labels: dict[str, dict[str, int]]

    min_label_text_length: int
    average_label_text_length: float
    max_label_text_length: int


class AbsTaskZeroShotClassification(AbsTask):
    """Abstract class for ZeroShotClassification tasks
    The similarity between an images and candidate text prompts, such as this is a dog/this is a cat.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        image: list of Image.Image
        labels: list of int
    """

    image_column_name: str = "image"
    label_column_name: str = "label"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ZeroShotClassificationDescriptiveStatistics:
        if hf_subset:
            imgs = self.dataset[hf_subset][split][self.image_column_name]
            labels = self.dataset[hf_subset][split][self.label_column_name]
        elif compute_overall:
            imgs, labels = [], []
            for hf_subset in self.metadata.eval_langs:
                imgs.extend(self.dataset[hf_subset][split][self.image_column_name])
                labels.extend(self.dataset[hf_subset][split][self.label_column_name])
        else:
            imgs = self.dataset[split][self.image_column_name]
            labels = self.dataset[split][self.label_column_name]

        num_samples = len(labels)
        unique_num_labels = len(set(labels))
        label_count = Counter(labels)

        img_widths, img_heights = [], []
        for img in imgs:
            width, height = img.size  # type: ignore
            img_heights.append(height)
            img_widths.append(width)

        candidate_labels_len = [len(c) for c in self.get_candidate_labels()]

        return ZeroShotClassificationDescriptiveStatistics(
            num_samples=num_samples,
            unique_num_labels=unique_num_labels,
            min_image_width=min(img_widths),
            average_image_width=sum(img_widths) / len(img_widths),
            max_image_width=max(img_widths),
            min_image_height=min(img_heights),
            average_image_height=sum(img_heights) / len(img_heights),
            max_image_height=max(img_heights),
            min_label_text_length=min(candidate_labels_len),
            average_label_text_length=sum(candidate_labels_len)
            / len(candidate_labels_len),
            max_label_text_length=max(candidate_labels_len),
            labels={
                str(label): {"count": count} for label, count in label_count.items()
            },
        )

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        candidate_labels = self.get_candidate_labels()
        evaluator = ZeroShotClassificationEvaluator(
            dataset,
            self.image_column_name,
            dataset[self.label_column_name],
            candidate_labels,
            task_name=self.metadata.name,
            **kwargs,
        )
        metrics = evaluator(model, encode_kwargs=encode_kwargs)

        scores = {"accuracy": metrics["accuracy"]}
        self._add_main_score(scores)
        return scores

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        raise NotImplementedError("This method should be overridden by subclasses")
