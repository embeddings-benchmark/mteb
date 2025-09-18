from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tqdm
from datasets import Dataset

from mteb._evaluators import ClusteringEvaluator
from mteb.models import Encoder
from mteb.types import ScoresDict
from mteb.types.statistics import (
    ImageStatistics,
    LabelStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

from ._statistics_calculation import (
    calculate_image_statistics,
    calculate_label_statistics,
    calculate_text_statistics,
)
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class ClusteringDescriptiveStatistics(SplitDescriptiveStatistics):
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
    """Abstract class for Clustering tasks for any modality.
    The similarity is computed between pairs and the results are ranked.
    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        input_column_name: list of inputs (str | image).
        label_column_name: list of str or class labels.
    """

    abstask_prompt = "Identify categories in user passages."
    evaluator: type[ClusteringEvaluator] = ClusteringEvaluator
    input_column_name: str = "sentences"
    label_column_name: str = "labels"

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
        # MTEB text clustering requires renaming and eval per subset.
        if self.metadata.modalities == ["text"]:
            v_measures = []
            for cluster_set in tqdm.tqdm(data_split, desc="Clustering"):
                clustering_dataset = Dataset.from_dict(cluster_set).select_columns(
                    [self.input_column_name, self.label_column_name]
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
                metrics = evaluator(
                    model, encode_kwargs=encode_kwargs, v_measure_only=True
                )
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

        data_split = data_split.select_columns(
            [self.input_column_name, self.label_column_name]
        )
        evaluator = self.evaluator(
            data_split,
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

    def _calculate_descriptive_statistics_from_split(
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

        if isinstance(inputs[0], list):
            inputs = [item for sublist in inputs for item in sublist]
        if isinstance(labels[0], list):
            labels = [item for sublist in labels for item in sublist]

        total_text_len = 0
        text_statistics, image_statistics = None, None
        if "image" in self.metadata.modalities:
            image_statistics = calculate_image_statistics(inputs)

        if "text" in self.metadata.modalities:
            text_statistics = calculate_text_statistics(inputs)

        label_statistics = calculate_label_statistics(labels)

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
