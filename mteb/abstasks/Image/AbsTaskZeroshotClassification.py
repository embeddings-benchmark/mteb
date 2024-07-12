from __future__ import annotations

import logging
from typing import Any

import tqdm
from datasets import Dataset

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.load_results.mteb_results import ScoresDict

from ...evaluation.evaluators import ZeroshotClassificationEvaluator
from ..AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskZeroshotClassification(AbsTask):
    """Abstract class for ZeroshotClassification tasks
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

    def _evaluate_subset(
        self,
        model: EncoderWithQueryCorpusEncode | Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:

        candidate_labels = self.get_candidate_labels()
        
        evaluator = ZeroshotClassificationEvaluator(
            dataset[self.image_column_name],  
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
