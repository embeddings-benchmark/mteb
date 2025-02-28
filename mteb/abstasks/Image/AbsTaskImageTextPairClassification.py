from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from ...encoder_interface import Encoder
from ...evaluation.evaluators import ImageTextPairClassificationEvaluator
from ..AbsTask import AbsTask, ScoresDict
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class ImageTextPairClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for ImageTextPairClassification

    Attributes:
        num_samples: number of samples in the dataset.
        num_images: number of images in the dataset.
        num_texts: number of texts in the dataset.
        num_unique_texts: number of unique texts in the dataset.

        min_text_length: Minimum length of texts
        average_text_length: Average length of texts
        max_text_length: Maximum length of texts
    """

    num_samples: int
    num_images: int
    num_texts: int
    num_unique_texts: int

    min_text_length: int
    average_text_length: float
    max_text_length: int


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
    ) -> ImageTextPairClassificationDescriptiveStatistics:
        dataset = (
            self.dataset[split] if hf_subset is None else self.dataset[hf_subset][split]
        )
        num_samples = len(dataset)

        if isinstance(self.images_column_names, str):
            num_images = list(dataset[self.images_column_names])
        elif isinstance(self.images_column_names, list):
            num_images = sum(
                [len(dataset[img_column]) for img_column in self.images_column_names]
            )

        if isinstance(self.texts_column_names, str):
            texts = list(dataset[self.texts_column_names])
            unique_texts = set(texts)
            text_lengths = [len(text) for text in texts]
        elif isinstance(self.texts_column_names, list):
            texts = [
                text
                for text_column in self.texts_column_names
                for text in dataset[text_column]
            ]
            unique_texts = set(texts)
            text_lengths = [len(text) for text in texts]

        return ImageTextPairClassificationDescriptiveStatistics(
            num_samples=num_samples,
            num_images=num_images,
            num_texts=len(texts),
            num_unique_texts=len(unique_texts),
            min_text_length=min(text_lengths),
            average_text_length=sum(text_lengths) / len(text_lengths),
            max_text_length=max(text_lengths),
        )

    def _evaluate_subset(
        self,
        model: Encoder,
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
