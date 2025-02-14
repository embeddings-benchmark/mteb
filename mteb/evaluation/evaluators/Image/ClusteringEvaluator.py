from __future__ import annotations

import logging
from typing import Any

import sklearn
import sklearn.cluster
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ImageClusteringEvaluator(Evaluator):
    def __init__(
        self,
        images: list[Image.Image],
        labels: list[int],
        task_name: str | None = None,
        clustering_batch_size: int = 500,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            images = images[:limit]
            labels = labels[:limit]
        self.images = images
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        image_embeddings = model.get_image_embeddings(
            self.images,
            batch_size=encode_kwargs["batch_size"],
        )

        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)),
            batch_size=self.clustering_batch_size,
            n_init="auto",
        )
        clustering_model.fit(image_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = metrics.cluster.v_measure_score(self.labels, cluster_assignment)
        nmi = metrics.cluster.normalized_mutual_info_score(
            self.labels, cluster_assignment
        )
        ari = metrics.cluster.adjusted_rand_score(self.labels, cluster_assignment)

        matrix = metrics.confusion_matrix(self.labels, cluster_assignment)

        # get linear sum assignment
        row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
        total_correct = matrix[row_ind, col_ind].sum()
        clustering_accuracy = total_correct / len(self.labels)

        return {
            "v_measure": v_measure,
            "nmi": nmi,
            "ari": ari,
            "cluster_accuracy": clustering_accuracy,
        }
