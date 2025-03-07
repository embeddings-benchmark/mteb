from __future__ import annotations

import logging
from typing import Any

import sklearn
import sklearn.cluster
from datasets import Dataset
from sklearn import metrics
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ClusteringEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        task_name: str | None = None,
        clustering_batch_size: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.clustering_batch_size = clustering_batch_size
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        corpus_embeddings = model.encode(
            DataLoader(self.dataset),
            task_name=self.task_name,
            **encode_kwargs,
        )

        labels = self.dataset["labels"]
        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(labels)),
            batch_size=self.clustering_batch_size,
            n_init="auto",
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = metrics.cluster.v_measure_score(labels, cluster_assignment)

        return {"v_measure": v_measure}
