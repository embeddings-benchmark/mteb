import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import torch
from datasets import Dataset, concatenate_datasets

from mteb._evaluators import ImageTextPairClassificationEvaluator
from mteb.abstasks._statistics_calculation import (
    calculate_image_statistics,
    calculate_text_statistics,
)
from mteb.abstasks.abstask import AbsTask
from mteb.models.models_protocols import EncoderProtocol
from mteb.types.statistics import (
    ImageStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

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


class ImageTextPairClassificationMetrics(TypedDict):
    """ImageTextPairClassification metrics.

    Attributes:
        image_acc: Accuracy of image retrieval.
        text_acc: Accuracy of text retrieval.
        accuracy: Overall accuracy.
    """

    image_acc: float
    text_acc: float
    accuracy: float


class AbsTaskImageTextPairClassification(AbsTask):
    """Abstract class for Image Text Pair Classification tasks (Compositionality evaluation).

    The similarity is computed between pairs and the results are ranked.
    Note that the number of images and the number of captions can be different.

    Attributes:
        dataset: A HuggingFace Dataset containing the data for the ImageTextPairClassification task. Should have columns:
            - images: List of images.
            - captions: List of captions.
        images_column_names: Name of the column(s) containing the images.
        texts_column_names: Name of the column(s) containing the captions.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
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
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> ImageTextPairClassificationMetrics:
        select_columns = []
        for columns in (self.images_column_names, self.texts_column_names):
            if isinstance(columns, str):
                select_columns.append(columns)
            else:
                select_columns.extend(columns)

        data_split = data_split.select_columns(select_columns)
        num_images_per_sample = (
            1
            if isinstance(self.images_column_names, str)
            else len(self.images_column_names)
        )
        num_texts_per_sample = (
            1
            if isinstance(self.texts_column_names, str)
            else len(self.texts_column_names)
        )
        evaluator = ImageTextPairClassificationEvaluator(
            data_split,
            images_column_names=self.images_column_names,
            texts_column_names=self.texts_column_names,
            task_metadata=self.metadata,
            num_texts_per_sample=num_texts_per_sample,
            num_images_per_sample=num_images_per_sample,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        if prediction_folder:
            self._save_task_predictions(
                [score.tolist() for score in scores],
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        return self._compute_metrics(
            scores,
            num_images_per_sample,
            num_texts_per_sample,
        )

    def _compute_metrics(
        self,
        scores: list[torch.Tensor],
        num_images_per_sample: int,
        num_texts_per_sample: int,
    ) -> ImageTextPairClassificationMetrics:
        image_score = []
        text_score = []
        all_correct_scores = []
        img_ground_truths = torch.arange(num_images_per_sample)
        caption_ground_truths = torch.arange(num_texts_per_sample)

        for score in scores:
            image_closest_text = score.argmax(dim=1)  # shape = (num_images_per_sample)
            text_closest_image = score.argmax(dim=0)  # shape = (num_texts_per_sample)
            pred_text_is_correct = (
                (image_closest_text == img_ground_truths).all().item()
            )
            pred_image_is_correct = (
                (text_closest_image == caption_ground_truths).all().item()
            )
            all_correct = pred_text_is_correct and pred_image_is_correct
            image_score.append(pred_image_is_correct)
            text_score.append(pred_text_is_correct)
            all_correct_scores.append(all_correct)

        return ImageTextPairClassificationMetrics(
            image_acc=torch.Tensor(image_score).float().mean().item(),
            text_acc=torch.Tensor(text_score).float().mean().item(),
            accuracy=torch.Tensor(all_correct_scores).float().mean().item(),
        )

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
