from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from ...encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from ...evaluation.evaluators import ImageTextPairClassificationEvaluator
from ...load_results.mteb_results import ScoresDict
from ..AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskImageTextPairClassification(AbsTask):
    """Abstract class for Image Text Pair Classification tasks,
    e.g. Compositionality evaluation.
    The similarity is computed between pairs and the results are ranked.
    Note that the number of images and the number of captions can be different.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        images: List[List[Image.Image]]
        captions: List[List[str]]
    """

    # it can be ["image_0", "image_1"]; ["text_0", "text_1"] for datasets like WinoGround
    images_column_names: str | list[str] = "image"
    texts_column_names: str | list[str] = "caption"

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
        model: Encoder | EncoderWithQueryCorpusEncode,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        evaluator = ImageTextPairClassificationEvaluator(
            dataset,
            images_column_names=self.images_column_names,
            texts_column_names=self.texts_column_names,
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(scores)
        return scores
