from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from datasets import Dataset

from ...encoder_interface import Encoder
from ...evaluation.evaluators import ZeroshotClassificationEvaluator
from ..AbsTask import AbsTask, ScoresDict
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class ZeroshotClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for ZeroshotClassification

    Attributes:
        num_samples: number of samples in the dataset.

        unique_labels: Number of unique labels
        labels: dict of label frequencies

        min_label_text_length: Minimum length of candidate label text
        average_label_text_length: Average length of candidate label text
        max_label_text_length: Maximum length of candidate label text
    """

    num_samples: int

    #! Check if these are actually valid
    unique_num_labels: int
    labels: dict[str, dict[str, int]]

    min_label_text_length: int
    average_label_text_length: float
    max_label_text_length: int


class AbsTaskZeroshotClassification(AbsTask):
    """Abstract class for ZeroshotClassification tasks
    The similarity between an image and candidate text prompts, such as this is a .

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        image: list of Image.Image
        labels: list of int
    """

    audio_column_name: str = "image"
    label_column_name: str = "label"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        pass

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ZeroshotClassificationDescriptiveStatistics:
        pass

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        pass

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        raise NotImplementedError("This method should be overridden by subclasses")
