from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import tqdm
from datasets import Dataset

from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import ScoresDict

from ..evaluation.evaluators import ClusteringEvaluator
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class ClusteringDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Clustering

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.

        min_text_length: Minimum length of text
        average_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_texts: Number of unique texts

        min_labels_per_text: Minimum number of labels per text
        average_labels_per_text: Average number of labels per text
        max_labels_per_text: Maximum number of labels per text
        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int
    number_of_characters: int

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_texts: int

    min_labels_per_text: int
    average_labels_per_text: float
    max_labels_per_text: int

    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskClustering(AbsTask):
    """Abstract class for Clustering tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sentences: list of str
        labels: list of str
    """

    abstask_prompt = "Identify categories in user passages."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        v_measures = []
        for cluster_set in tqdm.tqdm(dataset, desc="Clustering"):
            evaluator = ClusteringEvaluator(
                cluster_set["sentences"],  # type: ignore
                cluster_set["labels"],  # type: ignore
                task_name=self.metadata.name,
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
                sentences.extend(self.dataset[hf_subset][split]["sentences"])
                labels.extend(self.dataset[hf_subset][split]["labels"])
        else:
            sentences = self.dataset[split]["sentences"]
            labels = self.dataset[split]["labels"]

        text_len = [len(t) for t in sentences]
        all_sentences = []
        for s in sentences:
            all_sentences.extend(s)
        total_text_len = sum(text_len)
        total_labels = []
        for label in labels:
            if isinstance(label, list):
                total_labels.extend(label)
            else:
                total_labels.append(label)
        label_counter = Counter(total_labels)
        return ClusteringDescriptiveStatistics(
            num_samples=len(sentences),
            number_of_characters=total_text_len,
            min_text_length=min(text_len),
            average_text_length=total_text_len / len(sentences),
            max_text_length=max(text_len),
            unique_texts=len(set(all_sentences)),
            min_labels_per_text=min(label_counter.values()),
            average_labels_per_text=len(total_labels) / len(sentences),
            max_labels_per_text=max(label_counter.values()),
            unique_labels=len(label_counter),
            labels={
                str(label): {
                    "count": value,
                }
                for label, value in label_counter.items()
            },
        )
