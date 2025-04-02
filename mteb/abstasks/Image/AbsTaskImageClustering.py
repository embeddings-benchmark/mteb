from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from datasets import Dataset

from mteb.abstasks.TaskMetadata import HFSubset

from ...encoder_interface import Encoder
from ...evaluation.evaluators import ImageClusteringEvaluator
from ..AbsTask import AbsTask, ScoresDict
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class ImageClusteringDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for ImageClustering

    Attributes:
        num_samples: number of samples in the dataset.

        min_image_width: Minimum width of images
        average_image_width: Average width of images
        max_image_width: Maximum width of images

        min_image_height: Minimum height of images
        average_image_height: Average height of images
        max_image_height: Maximum height of images

        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int

    min_image_width: float
    average_image_width: float
    max_image_width: float

    min_image_height: float
    average_image_height: float
    max_image_height: float

    unique_num_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskImageClustering(AbsTask):
    """Abstract class for Clustering tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        image: Image.Image
        label: int
    """

    image_column_name: str = "image"
    label_column_name: str = "label"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores: dict[HFSubset, ScoresDict]) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ImageClusteringDescriptiveStatistics:
        if hf_subset:
            imgs = self.dataset[hf_subset][split][self.image_column_name]
            labels = self.dataset[hf_subset][split][self.label_column_name]
        elif compute_overall:
            imgs, labels = [], []
            for hf_subset in self.metadata.eval_langs:
                imgs.extend(self.dataset[hf_subset][split][self.image_column_name])
                labels.extend(self.dataset[hf_subset][split][self.label_column_name])
        else:
            imgs = self.dataset[split][self.image_column_name]
            labels = self.dataset[split][self.label_column_name]

        num_samples = len(labels)
        unique_num_labels = len(set(labels))
        label_count = Counter(labels)

        img_widths, img_heights = [], []
        for img in imgs:
            width, height = img.size  # type: ignore
            img_heights.append(height)
            img_widths.append(width)

        return ImageClusteringDescriptiveStatistics(
            num_samples=num_samples,
            unique_num_labels=unique_num_labels,
            min_image_width=min(img_widths),
            average_image_width=sum(img_widths) / len(img_widths),
            max_image_width=max(img_widths),
            min_image_height=min(img_heights),
            average_image_height=sum(img_heights) / len(img_heights),
            max_image_height=max(img_heights),
            labels={
                str(label): {"count": count} for label, count in label_count.items()
            },
        )

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        evaluator = ImageClusteringEvaluator(
            dataset[self.image_column_name],
            dataset[self.label_column_name],
            task_name=self.metadata.name,
            **kwargs,
        )
        metrics = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(metrics)
        return metrics
