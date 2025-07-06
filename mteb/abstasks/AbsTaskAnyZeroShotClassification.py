from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from datasets import Dataset

from mteb.types import ScoresDict
from mteb.types.statistics import (
    DescriptiveStatistics,
    ImageStatistics,
    LabelStatistics,
    TextStatistics,
)

from ..evaluation.evaluators import ZeroShotClassificationEvaluator
from ..models.encoder_interface import Encoder
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class ZeroShotClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for ZeroShotClassification

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: None (no text inputs)

        text_statistics: None (no text inputs)
        image_statistics: Statistics for images
        label_statistics: Statistics for dataset labels

        min_label_text_length: Minimum length of candidate label text
        average_label_text_length: Average length of candidate label text
        max_label_text_length: Maximum length of candidate label text
    """

    num_samples: int
    number_of_characters: int | None

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    label_statistics: LabelStatistics

    min_label_text_length: int
    average_label_text_length: float
    max_label_text_length: int


class AbsTaskAnyZeroShotClassification(AbsTask):
    """Abstract class for ZeroShot Classification tasks for any modality.
    The similarity between an image (or audio) and candidate text prompts, such as this is a dog/this is a cat.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        inputs: list of Image.Image or audio
        label: list of int
    """

    input_column_name: str = "image"
    label_column_name: str = "label"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ZeroShotClassificationDescriptiveStatistics:
        if hf_subset:
            inputs = self.dataset[hf_subset][split][self.input_column_name]
            labels = self.dataset[hf_subset][split][self.label_column_name]
        elif compute_overall:
            inputs, labels = [], []
            for hf_subset in self.metadata.eval_langs:
                inputs.extend(self.dataset[hf_subset][split][self.input_column_name])
                labels.extend(self.dataset[hf_subset][split][self.label_column_name])
        else:
            inputs = self.dataset[split][self.input_column_name]
            labels = self.dataset[split][self.label_column_name]

        num_samples = len(inputs)
        label_count = Counter(labels)

        # build image statistics
        img_widths, img_heights = [], []
        for img in inputs:
            w, h = img.size  # type: ignore
            img_widths.append(w)
            img_heights.append(h)

        image_statistics = ImageStatistics(
            min_image_width=min(img_widths),
            average_image_width=sum(img_widths) / len(img_widths),
            max_image_width=max(img_widths),
            min_image_height=min(img_heights),
            average_image_height=sum(img_heights) / len(img_heights),
            max_image_height=max(img_heights),
        )

        # single‐label per sample => use LabelStatistics
        label_statistics = LabelStatistics(
            min_labels_per_text=1,
            average_label_per_text=1.0,
            max_labels_per_text=1,
            unique_labels=len(label_count),
            labels={str(lbl): {"count": cnt} for lbl, cnt in label_count.items()},
        )

        # candidate‐label text lengths
        candidate_lens = [len(c) for c in self.get_candidate_labels()]

        return ZeroShotClassificationDescriptiveStatistics(
            num_samples=num_samples,
            number_of_characters=None,
            text_statistics=None,
            image_statistics=image_statistics,
            label_statistics=label_statistics,
            min_label_text_length=min(candidate_lens),
            average_label_text_length=sum(candidate_lens) / len(candidate_lens),
            max_label_text_length=max(candidate_lens),
        )

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ScoresDict:
        candidate_labels = self.get_candidate_labels()
        evaluator = ZeroShotClassificationEvaluator(
            dataset,
            self.input_column_name,
            dataset[self.label_column_name],
            candidate_labels,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        metrics = evaluator(model, encode_kwargs=encode_kwargs)

        scores = {"accuracy": metrics["accuracy"]}
        self._add_main_score(scores)
        return scores

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        raise NotImplementedError("This method should be overridden by subclasses")
