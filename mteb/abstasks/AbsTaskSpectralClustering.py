from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
import sklearn
import sklearn.cluster
import tqdm
from datasets import Dataset
from sklearn import metrics

from mteb.encoder_interface import Encoder

from ..evaluation.evaluators import Evaluator
from ..evaluation.evaluators.utils import cos_sim
from ..load_results.task_results import ScoresDict
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class SpectralClusteringEvaluator(Evaluator):
    def __init__(
        self,
        sentences,
        labels,
        task_name: str | None = None,
        clustering_batch_size: int = 500,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            sentences = sentences[:limit]
            labels = labels[:limit]
        self.sentences = sentences
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        corpus_embeddings = model.encode(
            self.sentences,
            task_name=self.task_name,
            **encode_kwargs,
        )

        ## Build a adjacency-matrix, which edge is similarity (defalut : cosine-similarity)
        logger.info("Building a graph model...")
        G = nx.Graph()
        for i, i_text in enumerate(self.sentences[:-1]):
            score_list = (
                model.similarity(corpus_embeddings[i], corpus_embeddings[i + 1 :])[0]
                * 100
                if getattr(model, "similarity", None)
                else cos_sim(corpus_embeddings[i], corpus_embeddings[i + 1 :])[0] * 100
            )
            for j_text, score in zip(self.sentences[i + 1 :], score_list):
                G.add_edge(i_text, j_text, weight=score)

        ## Convert to numpy array, and Negative values are replaced with 0
        adjacency_cos_score_matrix = nx.to_numpy_array(G)
        adjacency_cos_score_matrix = np.where(
            adjacency_cos_score_matrix < 0, 0, adjacency_cos_score_matrix
        )

        ## Spectral Clustering
        clustering = sklearn.cluster.SpectralClustering(
            n_clusters=len(set(self.labels)),
            affinity="precomputed",
            assign_labels="discretize",
        )
        clustering.fit(adjacency_cos_score_matrix)
        cluster_assignment = clustering.labels_

        logger.info("Evaluating...")
        v_measure = metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}


class AbsTaskSpectralClustering(AbsTask):
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
            evaluator = SpectralClusteringEvaluator(
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
