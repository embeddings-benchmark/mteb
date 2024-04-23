from __future__ import annotations

import logging
import random
from typing import Any, Dict

import numpy as np
import sklearn
import sklearn.cluster
from datasets import DatasetDict
from sklearn.metrics.cluster import v_measure_score

from .AbsTask import AbsTask

logger = logging.getLogger(__name__)

Split = str
HFLang = str
MultilingualDataset = Dict[HFLang, DatasetDict]
Scores = Dict[str, Any]


def evaluate_clustering_bootstrapped(
    embeddings: np.ndarray,
    labels: list[str],
    n_clusters: int,
    cluster_size: int,
    kmean_batch_size: int,
    rng_state: random.Random = random.Random(),
) -> list[float]:
    """Bootstrapped evaluation of clustering performance using V-measure.

    The bootstrapping is done by sampling N samples from the corpus and clustering them. It is done without replacement to get a diverse set of
    samples.
    """
    n_embeddings = embeddings.shape[0]
    labels_arr = np.array(labels)

    v_measures = []

    clustering_model = sklearn.cluster.MiniBatchKMeans(
        n_clusters=len(set(labels)),
        batch_size=kmean_batch_size,
        n_init="auto",
    )

    for _ in range(n_clusters):
        # sample N samples from the corpus with replacement
        cluster_indices = rng_state.choices(range(n_embeddings), k=cluster_size)

        _embeddings = embeddings[cluster_indices]
        _labels = labels_arr[cluster_indices]
        clustering_model.fit(_embeddings)
        cluster_assignment = clustering_model.labels_
        v_measure = v_measure_score(_labels, cluster_assignment)
        v_measures.append(v_measure)

    return v_measures


class AbsTaskClusteringFast(AbsTask):
    """Abstract class for Clustering tasks.

    This class embeds the corpus sentences then samples N samples from the corpus and clusters them.
    The similarity then is calculated using the V-measure metric, which is invariant to the permutation of the labels.
    This approach is then repeated K times.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset.
    It must contain the following columns:
        sentences: list[str]
        labels: list[str]
    """

    max_documents_to_embed = 16_384
    max_documents_per_cluster = 2048
    n_clusters = 10
    k_mean_batch_size = 512

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores):
        if self.metadata_dict["main_score"] in scores:
            scores["main_score"] = scores[self.metadata_dict["main_score"]]
        else:
            logger.warn(
                f"main score {self.metadata_dict['main_score']} not found in scores {scores.keys()}"
            )

    def evaluate(self, model, split="test", **kwargs) -> Scores | Dict[HFLang, Scores]:
        lang: HFLang
        multilingual_ds: MultilingualDataset
        self.dataset: MultilingualDataset | DatasetDict
        ds: DatasetDict

        if not self.data_loaded:
            self.load_data()

        if self.is_multilingual:
            multilingual_ds = self.dataset  # type: ignore

            multilingual_scores: dict[HFLang, Scores] = {}
            for lang in self.dataset:  # type: ignore
                logger.info(
                    f"\nTask: {self.metadata.name}, split: {split}, language: {lang}. Running..."
                )
                _ds = multilingual_ds[lang]
                multilingual_scores[lang] = self._evaluate_monolingual(
                    model, multilingual_ds[lang], split, **kwargs
                )
                return multilingual_scores
        logger.info(f"\nTask: {self.metadata.name}, split: {split}. Running...")

        ds = self.dataset  # type: ignore
        scores = self._evaluate_monolingual(model, ds, split, **kwargs)
        return scores

    def _evaluate_monolingual(
        self, model, dataset: DatasetDict, split: Split = "test", **kwargs: Any
    ) -> dict[str, float | list[float]]:
        _dataset = dataset[split]

        rng_state = random.Random(self.seed)
        example_indices = rng_state.sample(
            range(len(_dataset)), k=self.max_documents_to_embed
        )
        downsampled_dataset = _dataset.select(example_indices)

        logger.info(f"Encoding {len(downsampled_dataset)} sentences...")

        embeddings = model.encode(downsampled_dataset["sentences"])

        v_measures = evaluate_clustering_bootstrapped(
            embeddings,
            downsampled_dataset["labels"],
            n_clusters=self.n_clusters,
            cluster_size=self.max_documents_per_cluster,
            kmean_batch_size=self.k_mean_batch_size,
            rng_state=rng_state,
        )

        return {"v_measures": v_measures, "v_measure": float(np.mean(v_measures))}
