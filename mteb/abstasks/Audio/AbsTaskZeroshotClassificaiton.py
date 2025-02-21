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
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        pass

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:

        candidate_labels = self.get_candidate_labels()
        evaluator = ZeroshotClassificationEvaluator(
            dataset,
            self.audio_column_name,
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
