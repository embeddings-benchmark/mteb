from __future__ import annotations

import itertools
import logging
import random
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import sklearn
import sklearn.cluster
from datasets import Dataset, DatasetDict
from sklearn.metrics.cluster import v_measure_score

from ..MTEBResults import HFSubset
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


MultilingualDataset = Dict[HFSubset, DatasetDict]


def evaluate_clustering_bootstrapped(
    embeddings: np.ndarray,
    labels: list[list[str]],
    n_clusters: int,
    cluster_size: int,
    kmean_batch_size: int,
    max_depth: Optional[int],
    rng_state: random.Random = random.Random(),
) -> dict[str, list[float]]:
    """Bootstrapped evaluation of clustering performance using V-measure.

    The bootstrapping is done by sampling N samples from the corpus and clustering them. It is done without replacement to get a diverse set of
    samples.
    """
    n_embeddings = embeddings.shape[0]

    v_measures = defaultdict(list)
    if max_depth is not None:
        max_depth = min(max_depth, max(map(len, labels)))
    else:
        max_depth = max(map(len, labels))
    # Evaluate on each level til max depth
    for i_level in range(max_depth):
        level_labels = []
        # Assign -1 to gold label if the level is not there
        for label in labels:
            if len(label) > i_level:
                level_labels.append(label[i_level])
            else:
                level_labels.append(-1)
        level_labels = np.array(level_labels)
        valid_idx = level_labels != -1
        level_labels = level_labels[valid_idx]
        level_embeddings = embeddings[valid_idx]
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(level_labels)),
            batch_size=kmean_batch_size,
            n_init="auto",
        )
        for _ in range(n_clusters):
            # sample N samples from the corpus with replacement
            n_embeddings = len(level_embeddings)
            cluster_indices = rng_state.choices(range(n_embeddings), k=cluster_size)

            _embeddings = level_embeddings[cluster_indices]
            _labels = level_labels[cluster_indices]
            cluster_assignment = clustering_model.fit_predict(_embeddings)
            v_measure = v_measure_score(_labels, cluster_assignment)
            v_measures[f"Level {i_level}"].append(v_measure)

    return v_measures


class AbsTaskClusteringFast(AbsTask):
    """Abstract class for Clustering tasks.

    This class embeds the corpus sentences then samples N samples from the corpus and clusters them.
    The similarity then is calculated using the V-measure metric, which is invariant to the permutation of the labels.
    This approach is then repeated K times.

    If the clustering is hieararchical, and more than one label is specified in order for each observation,
    V-measures are calculated in the outlined way on each of the levels separately.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset.
    It must contain the following columns:
        sentences: list[str]
        labels: list[str] | list[list[str]]
    """

    max_documents_to_embed = 16_384
    max_documents_per_cluster = 2048
    n_clusters = 10
    k_mean_batch_size = 512
    max_depth = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores):
        if self.metadata_dict["main_score"] in scores:
            scores["main_score"] = scores[self.metadata_dict["main_score"]]
        else:
            logger.warning(
                f"main score {self.metadata_dict['main_score']} not found in scores {scores.keys()}"
            )

    def _evaluate_subset(
        self, model, dataset: DatasetDict, **kwargs: Any
    ) -> dict[str, float | dict[str, list[float]]]:
        rng_state = random.Random(self.seed)

        if len(dataset) > self.max_documents_to_embed:
            example_indices = rng_state.sample(
                range(len(dataset)), k=self.max_documents_to_embed
            )
            downsampled_dataset = dataset.select(example_indices)
        else:
            downsampled_dataset = dataset

        logger.info(f"Encoding {len(downsampled_dataset)} sentences...")

        embeddings = model.encode(downsampled_dataset["sentences"])
        labels = []
        for label in downsampled_dataset["labels"]:
            if not isinstance(label, list):
                label = [label]
            labels.append(label)

        v_measures = evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=self.n_clusters,
            cluster_size=self.max_documents_per_cluster,
            kmean_batch_size=self.k_mean_batch_size,
            max_depth=self.max_depth,
            rng_state=rng_state,
        )
        all_v_scores = itertools.chain.from_iterable(v_measures.values())
        mean_v_measure = np.mean(list(all_v_scores))
        scores = {"v_measures": v_measures, "v_measure": float(mean_v_measure)}
        self._add_main_score(scores)
        return scores


def clustering_downsample(
    dataset: DatasetDict, seed: int, max_samples_in_cluster: int = 2048
) -> DatasetDict:
    """In cases where it is not possible to convert the dataset to a fast version, we can downsample the dataset to speed up the evaluation.

    This might be necessary when the clusters in the dataset is not sampled from the same distribution.
    """
    rng_state = random.Random(seed)

    ds = {}
    for split in dataset:
        _docs = []
        _labels = []

        n_clusters = len(dataset[split])

        for i in range(n_clusters):
            labels = dataset[split]["labels"][i]
            sentences = dataset[split]["sentences"][i]

            n_sample = min(max_samples_in_cluster, len(sentences))

            # sample n_sample from each cluster
            idxs = rng_state.sample(range(len(sentences)), n_sample)
            _docs.append([sentences[idx] for idx in idxs])
            _labels.append([labels[idx] for idx in idxs])

        ds[split] = Dataset.from_dict({"sentences": _docs, "labels": _labels})
    return DatasetDict(ds)


def convert_to_fast(
    dataset: DatasetDict, seed: int, max_size: int = 100_000
) -> DatasetDict:
    """Converts a clustering dataset to a fast version. This concats the cluster into two columns, sentences and labels.
    It additionally downsamples the dataset to max_size.
    """
    rng_state = random.Random(seed)

    ds = {}
    for split in dataset:
        sent_set = set()
        labels = []
        sentences = []
        n_clusters = len(dataset[split])
        all_labels_set = set(itertools.chain.from_iterable(dataset[split]["labels"]))
        for i in range(n_clusters):
            lab = dataset[split]["labels"][i]
            sents = dataset[split]["sentences"][i]

            # check that it is the same distribution
            row_label_set = set(lab)
            assert row_label_set.issubset(
                all_labels_set
            ), "The clusters are not sampled from the same distribution as they have different labels."

            for l, s in zip(lab, sents):
                if s not in sent_set:
                    labels.append(l)
                    sentences.append(s)
                    sent_set.add(s)  # ensuring no duplicates

        ds[split] = Dataset.from_dict({"sentences": sentences, "labels": labels})

        if len(ds[split]) > max_size:
            idxs = rng_state.sample(range(len(ds[split])), max_size)
            ds[split] = ds[split].select(idxs)

    return DatasetDict(ds)


def check_label_distribution(ds: DatasetDict) -> None:
    """For older clustering dataset versions.
    ds is a DatasetDict at the split level
    """
    n_clusters = len(ds)
    if n_clusters > 50:
        return
    all_labels_set = set(itertools.chain.from_iterable(ds["labels"]))

    for i in range(n_clusters):
        lab = ds["labels"][i]

        # check that it is the same distribution
        row_label_set = set(lab)
        assert row_label_set.issubset(
            all_labels_set
        ), "The clusters are not sampled from the same distribution as they have different labels."
