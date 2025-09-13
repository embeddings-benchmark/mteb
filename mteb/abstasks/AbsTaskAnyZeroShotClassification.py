from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from mteb._evaluators import ZeroShotClassificationEvaluator
from mteb.types import ScoresDict
from mteb.types.statistics import (
    ImageStatistics,
    LabelStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

from ..models import Encoder
from ._statistics_calculation import (
    calculate_image_statistics,
    calculate_label_statistics,
    calculate_text_statistics,
)
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class ZeroShotClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for ZeroShotClassification

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: None (no text inputs)

        text_statistics: None (no text inputs)
        image_statistics: Statistics for images
        label_statistics: Statistics for dataset labels

        candidates_labels_text_statistics: Statistics for candidate labels text
    """

    num_samples: int
    number_of_characters: int | None

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    label_statistics: LabelStatistics
    candidates_labels_text_statistics: TextStatistics


class AbsTaskAnyZeroShotClassification(AbsTask):
    """Abstract class for ZeroShot Classification tasks for any modality.
    The similarity between an image (or audio) and candidate text prompts, such as this is a dog/this is a cat.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        inputs: list of Image.Image or audio
        label: list of int
    """

    input_column_name: str = "image"
    label_column_name: str = "label"

    def _calculate_descriptive_statistics_from_split(
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

        image_statistics = None
        text_statistics = None

        if "image" in self.metadata.modalities:
            image_statistics = calculate_image_statistics(inputs)
        if self.metadata.modalities == ["text"]:
            text_statistics = calculate_text_statistics(inputs)

        label_statistics = calculate_label_statistics(labels)
        candidate_lens = calculate_text_statistics(self.get_candidate_labels())

        return ZeroShotClassificationDescriptiveStatistics(
            num_samples=num_samples,
            number_of_characters=None,
            text_statistics=text_statistics,
            image_statistics=image_statistics,
            label_statistics=label_statistics,
            candidates_labels_text_statistics=candidate_lens,
        )

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ScoresDict:
        candidate_labels = self.get_candidate_labels()
        evaluator = ZeroShotClassificationEvaluator(
            data_split,
            self.input_column_name,
            self.label_column_name,
            candidate_labels,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        return evaluator(model, encode_kwargs=encode_kwargs)

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        raise NotImplementedError("This method should be overridden by subclasses")
