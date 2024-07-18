from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from mteb.abstasks import AbsTask
from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.evaluation.evaluators import ImageTextPairClassificationEvaluator
from mteb.load_results.mteb_results import ScoresDict

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

    images_column_name: str = "images"
    texts_column_name: str = "captions"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: Encoder | EncoderWithQueryCorpusEncode,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        evaluator = ImageTextPairClassificationEvaluator(
            dataset[self.images_column_name],
            dataset[self.texts_column_name],
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(scores)
        return scores
