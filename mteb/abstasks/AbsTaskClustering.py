from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tqdm
from datasets import Dataset

from mteb.encoder_interface import Encoder
from mteb.types import ScoresDict
from mteb.types.statistics import DescriptiveStatistics, LabelStatistics, TextStatistics

from ..evaluation.evaluators import ClusteringEvaluator
from .AbsTask import AbsTask
from .statistics_calculation import (
    calculate_label_statistics,
    calculate_text_statistics,
)

logger = logging.getLogger(__name__)


class ClusteringDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Clustering

    Attributes:
        num_samples: number of samples in the dataset.

        text_statistics: Statistics for the text
        labels_statistics: Statistics for the labels
    """

    num_samples: int

    text_statistics: TextStatistics
    labels_statistics: LabelStatistics


class AbsTaskClustering(AbsTask):
    """Abstract class for Clustering tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        sentences: list of str
        labels: list of str
    """

    abstask_prompt = "Identify categories in user passages."

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
        v_measures = []
        for cluster_set in tqdm.tqdm(dataset, desc="Clustering"):
            clustering_dataset = Dataset.from_dict(cluster_set).rename_column(
                original_column_name="sentences", new_column_name="text"
            )
            evaluator = ClusteringEvaluator(
                clustering_dataset,
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                **kwargs,
            )
            metrics = evaluator(model, encode_kwargs=encode_kwargs)
            v_measures.append(metrics["v_measure"])

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        scores = {"v_measure": v_mean, "v_measure_std": v_std, "v_measures": v_measures}
        self._add_main_score(scores)
        return scores

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClusteringDescriptiveStatistics:
        if hf_subset:
            sentences = self.dataset[hf_subset][split]["sentences"]
            labels = self.dataset[hf_subset][split]["labels"]
        elif compute_overall:
            sentences = []
            labels = []
            for hf_subset in self.metadata.eval_langs:
                cur_sentences = self.dataset[hf_subset][split]["sentences"]
                cur_labels = self.dataset[hf_subset][split]["labels"]
                if len(cur_sentences) == 1 and len(cur_labels) == 1:
                    cur_sentences = cur_sentences[0]
                    cur_labels = cur_labels[0]
                sentences.extend(cur_sentences)
                labels.extend(cur_labels)
        else:
            sentences = self.dataset[split]["sentences"]
            labels = self.dataset[split]["labels"]

        if len(sentences) == 1 and len(labels) == 1:
            sentences = sentences[0]
            labels = labels[0]

        return ClusteringDescriptiveStatistics(
            num_samples=len(sentences),
            text_statistics=calculate_text_statistics(sentences),
            labels_statistics=calculate_label_statistics(labels),
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self._upload_dataset_to_hub(repo_name, ["sentences", "labels"])
