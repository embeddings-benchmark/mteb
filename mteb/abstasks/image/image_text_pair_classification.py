import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets

from mteb._evaluators import ImageTextPairClassificationEvaluator
from mteb.models.models_protocols import Encoder
from mteb.types import ScoresDict

from ...types.statistics import (
    ImageStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)
from .._statistics_calculation import (
    calculate_image_statistics,
    calculate_text_statistics,
)
from ..abstask import AbsTask

logger = logging.getLogger(__name__)


class ImageTextPairClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for ImageTextPairClassification

    Attributes:
        num_samples: number of samples in the dataset.
        text_statistics: Statistics for text
        image_statistics: Statistics for images
    """

    num_samples: int
    text_statistics: TextStatistics
    image_statistics: ImageStatistics


class AbsTaskImageTextPairClassification(AbsTask):
    """Abstract class for Image Text Pair Classification tasks,
    e.g. Compositionality evaluation.
    The similarity is computed between pairs and the results are ranked.
    Note that the number of images and the number of captions can be different.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        images: List[List[Image.Image]]
        captions: List[List[str]]
    """

    # it can be ["image_0", "image_1"]; ["text_0", "text_1"] for datasets like WinoGround
    images_column_names: str | Sequence[str] = "image"
    texts_column_names: str | Sequence[str] = "caption"

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ImageTextPairClassificationDescriptiveStatistics:
        if compute_overall:
            dataset = concatenate_datasets(
                [
                    self.dataset[hf_subset][split]
                    for hf_subset in self.metadata.eval_langs
                ]
            )
        else:
            dataset = (
                self.dataset[split]
                if hf_subset is None
                else self.dataset[hf_subset][split]
            )
        num_samples = len(dataset)

        images = None
        texts = None

        if isinstance(self.images_column_names, str):
            images = list(dataset[self.images_column_names])
        elif isinstance(self.images_column_names, Sequence):
            images = [
                img
                for img_column in self.images_column_names
                for img in dataset[img_column]
            ]

        if isinstance(self.texts_column_names, str):
            texts = list(dataset[self.texts_column_names])
        elif isinstance(self.texts_column_names, Sequence):
            texts = [
                text
                for text_column in self.texts_column_names
                for text in dataset[text_column]
            ]

        return ImageTextPairClassificationDescriptiveStatistics(
            num_samples=num_samples,
            text_statistics=calculate_text_statistics(texts),
            image_statistics=calculate_image_statistics(images),
        )

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> ScoresDict:
        select_columns = []
        for columns in (self.images_column_names, self.texts_column_names):
            if isinstance(columns, str):
                select_columns.append(columns)
            else:
                select_columns.extend(columns)

        data_split = data_split.select_columns(select_columns)
        evaluator = ImageTextPairClassificationEvaluator(
            data_split,
            images_column_names=self.images_column_names,
            texts_column_names=self.texts_column_names,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        return scores

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        text_columns = (
            [self.texts_column_names]
            if isinstance(self.texts_column_names, str)
            else self.texts_column_names
        )
        image_columns = (
            [self.images_column_names]
            if isinstance(self.images_column_names, str)
            else self.images_column_names
        )

        self._upload_dataset_to_hub(
            repo_name,
            [*text_columns, *image_columns],
        )
