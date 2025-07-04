from __future__ import annotations

import logging
from typing import Any

import sklearn
import sklearn.cluster
from datasets import Dataset
from sklearn import metrics
import torch
from torch.utils.data import DataLoader

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.encoder_interface import Encoder

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ClusteringEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        clustering_batch_size: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.clustering_batch_size = clustering_batch_size
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any]):
        corpus_embeddings = model.encode(
            DataLoader(self.dataset),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        labels = self.dataset["labels"]
        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(labels)),
            batch_size=self.clustering_batch_size,
            n_init="auto",
        )
        # TODO: refactor:
        if isinstance(corpus_embeddings, torch.Tensor) and corpus_embeddings.is_sparse:
            corpus_embeddings = corpus_embeddings.to_dense()
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = metrics.cluster.v_measure_score(labels, cluster_assignment)

        return {"v_measure": v_measure}
