from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from datasets import Dataset

from mteb.encoder_interface import Encoder
from mteb.types import ScoresDict
from mteb.types.statistics import (
    DescriptiveStatistics,
    ImageStatistics,
    LabelStatistics,
    TextStatistics,
)

from ..evaluation.evaluators import ClusteringEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class ClusteringDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Clustering

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.

        text_statistics: Statistics for text
        image_statistics: Statistics for images
        label_statistics: Statistics for labels
    """

    num_samples: int
    number_of_characters: int | None

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    label_statistics: LabelStatistics


class AbsTaskAnyClustering(AbsTask):
    """Abstract class for Clustering tasks
    The similarity is computed between pairs and the results are ranked.
    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        sentences: list of str
        labels: list of str
    """

    abstask_prompt = "Identify categories in user passages."
    evaluator: type[ClusteringEvaluator] = ClusteringEvaluator
    input_column_name: str = "sentences"
    label_column_name: str = "labels"

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ScoresDict:
        ## MTEB v1 text clustering requires renaming and eval per subset.
        if "sentences" in dataset.column_names:
            v_measures = []
            for cluster_set in tqdm.tqdm(dataset, desc="Clustering"):
                clustering_dataset = Dataset.from_dict(cluster_set).rename_column(
                    original_column_name="sentences", new_column_name="text"
                )
                evaluator = self.evaluator(
                    clustering_dataset,
                    input_column_name=self.input_column_name,
                    label_column_name=self.label_column_name,
                    task_metadata=self.metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    **kwargs,
                )
                metrics = evaluator(model, encode_kwargs=encode_kwargs)
                v_measures.append(metrics["v_measure"])

            v_mean = np.mean(v_measures)
            v_std = np.std(v_measures)
            scores = {
                "v_measure": v_mean,
                "v_measure_std": v_std,
                "v_measures": v_measures,
            }
            self._add_main_score(scores)
            return scores

        evaluator = self.evaluator(
            dataset,
            input_column_name=self.input_column_name,
            label_column_name=self.label_column_name,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        metrics = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(metrics)
        return metrics

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClusteringDescriptiveStatistics:
        if hf_subset:
            inputs = self.dataset[hf_subset][split][self.input_column_name]
            labels = self.dataset[hf_subset][split][self.label_column_name]
        elif compute_overall:
            inputs = []
            labels = []
            for hf_subset in self.metadata.eval_langs:
                inputs.extend(self.dataset[hf_subset][split][self.input_column_name])
                labels.extend(self.dataset[hf_subset][split][self.label_column_name])
        else:
            inputs = self.dataset[split][self.input_column_name]
            labels = self.dataset[split][self.label_column_name]

        total_text_len = 0
        text_len = None
        img_widths, img_heights = None, None

        if "image" in self.metadata.modalities:
            img_widths, img_heights = [], []
            for img in inputs:
                width, height = img.size  # type: ignore
                img_heights.append(height)
                img_widths.append(width)
        if "text" in self.metadata.modalities:
            text_len = [len(t) for t in inputs]
            total_text_len = sum(text_len)

        text_statistics, image_statistics = None, None
        if text_len:
            text_statistics = TextStatistics(
                min_text_length=min(text_len),
                average_text_length=total_text_len / len(inputs),
                max_text_length=max(text_len),
                unique_texts=len(set(inputs)),
            )
        if img_widths:
            image_statistics = ImageStatistics(
                min_image_width=min(img_widths),
                average_image_width=sum(img_widths) / len(img_widths),
                max_image_width=max(img_widths),
                min_image_height=min(img_heights),
                average_image_height=sum(img_heights) / len(img_heights),
                max_image_height=max(img_heights),
            )

        # labels
        if isinstance(labels[0], int):
            label_len = [1] * len(labels)
            total_label_len = len(labels)
            total_labels = labels
        else:
            # multilabel case
            label_len = [len(l) for l in labels]
            total_label_len = sum(label_len)
            total_labels = []
            for l in labels:
                total_labels.extend(l if len(l) > 0 else [None])
        for label in labels:
            if isinstance(label, list):
                total_labels.extend(label)
            else:
                total_labels.append(label)
        label_count = Counter(total_labels)
        label_statistics = LabelStatistics(
            min_labels_per_text=min(label_len),
            average_label_per_text=total_label_len / len(labels),
            max_labels_per_text=max(label_len),
            unique_labels=len(label_count),
            labels={
                str(label): {
                    "count": value,
                }
                for label, value in label_count.items()
            },
        )
        return ClusteringDescriptiveStatistics(
            num_samples=len(inputs),
            number_of_characters=total_text_len,
            text_statistics=text_statistics,
            image_statistics=image_statistics,
            label_statistics=label_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self._upload_dataset_to_hub(
            repo_name,
            [
                self.input_column_name,
                self.label_column_name,
            ],
        )
