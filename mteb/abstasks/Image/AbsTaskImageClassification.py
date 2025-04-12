from __future__ import annotations

import logging
from collections import Counter, defaultdict

import numpy as np
from datasets import Dataset
from PIL import ImageFile

from mteb.abstasks.TaskMetadata import DescriptiveStatistics

from ...evaluation import logRegClassificationEvaluator
from ..AbsTaskClassification import AbsClassification

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


class ImageClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for ImageClassification

    Attributes:
        num_samples: number of samples in the dataset.

        min_image_width: Minimum width of images
        average_image_width: Average width of images
        max_image_width: Maximum width of images

        min_image_height: Minimum height of images
        average_image_height: Average height of images
        max_image_height: Maximum height of images

        unique_num_labels: Number of unique labels
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


class AbsTaskImageClassification(AbsClassification):
    """Abstract class for kNN classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It
    must contain the following columns:
        image: Image.Image
        label: int
    """

    evaluator = logRegClassificationEvaluator
    values_column_name: str = "image"
    samples_per_label: int = 16
    n_experiments: int = 5
    abstask_prompt = "Classify user passages."
    is_image = True

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ImageClassificationDescriptiveStatistics:
        if hf_subset:
            imgs = self.dataset[hf_subset][split][self.values_column_name]
            labels = self.dataset[hf_subset][split][self.label_column_name]
        elif compute_overall:
            imgs, labels = [], []
            for hf_subset in self.metadata.eval_langs:
                imgs.extend(self.dataset[hf_subset][split][self.values_column_name])
                labels.extend(self.dataset[hf_subset][split][self.label_column_name])
        else:
            imgs = self.dataset[split][self.values_column_name]
            labels = self.dataset[split][self.label_column_name]

        num_samples = len(labels)
        unique_num_labels = len(set(labels))
        label_count = Counter(labels)

        img_widths, img_heights = [], []
        for img in imgs:
            width, height = img.size  # type: ignore
            img_heights.append(height)
            img_widths.append(width)

        return ImageClassificationDescriptiveStatistics(
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

    def _undersample_data(self, dataset_split: Dataset, idxs: list[int] | None = None):
        """Undersample data to have samples_per_label samples of each label
        without loading all images into memory.
        """
        if idxs is None:
            idxs = np.arange(len(dataset_split))
        self.np_rng.shuffle(idxs)
        if not isinstance(idxs, list):
            idxs = idxs.tolist()
        label_counter = defaultdict(int)
        selected_indices = []

        labels = dataset_split[self.label_column_name]
        for i in idxs:
            label = labels[i]
            if label_counter[label] < self.samples_per_label:
                selected_indices.append(i)
                label_counter[label] += 1

        undersampled_dataset = dataset_split.select(selected_indices)
        return (
            undersampled_dataset,
            idxs,
        )
