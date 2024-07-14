from __future__ import annotations

import logging
from typing import Any

import sklearn
import sklearn.cluster
from sklearn import metrics

from mteb.encoder_interface import Encoder

from .Evaluator import Evaluator
from .model_encode import model_encode

logger = logging.getLogger(__name__)


class ClusteringEvaluator(Evaluator):
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

        corpus_embeddings = model_encode(
            self.sentences,
            model=model,
            prompt_name=self.task_name,
            **encode_kwargs,
        )

        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)),
            batch_size=self.clustering_batch_size,
            n_init="auto",
        )
        clustering_model.fit(corpus_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = metrics.cluster.v_measure_score(self.labels, cluster_assignment)

        return {"v_measure": v_measure}
