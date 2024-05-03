from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
from datasets import DatasetDict
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import v_measure_score

from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


def evaluate_hierarchical_clustering(
    embeddings: np.ndarray,
    labels: list[list[str]],
    kmean_batch_size: int = 512,
    max_depth: int = 5,
    seed: int = 42,
) -> list[float]:
    """Evaluates clustering with v_scores on each level of the cluster hierarchy.
    On each level (up until the maximum depth) every example is evaluated that has that level.
    """
    v_scores = []
    max_depth = min(max_depth, max(map(len, labels)))
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
        clustering_model = MiniBatchKMeans(
            n_clusters=len(set(level_labels)),
            batch_size=kmean_batch_size,
            n_init="auto",
            random_state=seed,
        )
        pred_labels = clustering_model.fit_predict(level_embeddings)
        # Only evaluate where the level is valid
        score = v_measure_score(level_labels, pred_labels)
        v_scores.append(score)
    return v_scores


class AbsTaskHierarchicalClustering(AbsTask):
    """Abstract class for Hiearchical Clustering tasks."""

    max_documents_to_embed = 16_384
    k_mean_batch_size = 512
    max_depth = 5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores):
        if self.metadata_dict["main_score"] in scores:
            scores["main_score"] = scores[self.metadata_dict["main_score"]]
        else:
            logger.warn(
                f"main score {self.metadata_dict['main_score']} not found in scores {scores.keys()}"
            )

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        if self.is_multilingual:
            multilingual_ds = self.dataset

            multilingual_scores = {}
            for lang in self.dataset:
                logger.info(
                    f"\nTask: {self.metadata.name}, split: {split}, language: {lang}. Running..."
                )
                multilingual_scores[lang] = self._evaluate_monolingual(
                    model, multilingual_ds[lang], split, **kwargs
                )
                return multilingual_scores

        logger.info(f"\nTask: {self.metadata.name}, split: {split}. Running...")
        scores = self._evaluate_monolingual(model, self.dataset, split, **kwargs)
        return scores

    def _evaluate_monolingual(
        self, model, dataset: DatasetDict, split="test", **kwargs: Any
    ) -> dict[str, float | list[float]]:
        _dataset = dataset[split]

        rng_state = random.Random(self.seed)

        if len(_dataset) > self.max_documents_to_embed:
            example_indices = rng_state.sample(
                range(len(_dataset)), k=self.max_documents_to_embed
            )
            downsampled_dataset = _dataset.select(example_indices)
        else:
            downsampled_dataset = _dataset

        logger.info(f"Encoding {len(downsampled_dataset)} sentences...")
        embeddings = model.encode(downsampled_dataset["sentences"])
        v_measures = evaluate_hierarchical_clustering(
            embeddings,
            downsampled_dataset["labels"],
            kmean_batch_size=self.k_mean_batch_size,
            seed=self.seed,
        )
        return {"v_measures": v_measures, "v_measure": float(np.mean(v_measures))}
