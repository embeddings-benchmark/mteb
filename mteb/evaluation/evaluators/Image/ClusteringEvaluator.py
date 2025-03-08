from __future__ import annotations

import logging
from typing import Any

import sklearn
import sklearn.cluster
from PIL import Image
from datasets import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torch.utils.data import DataLoader

from mteb.create_dataloaders import prepare_image_dataset
from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ImageClusteringEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
            image_column_name: str,
            label_column_name: str,
        task_name: str | None = None,
        clustering_batch_size: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = prepare_image_dataset(
            dataset, image_column_name
        )
        self.labels = dataset[label_column_name]
        self.clustering_batch_size = clustering_batch_size
        self.task_name = task_name

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        image_embeddings = model.encode(
            DataLoader(
                self.dataset,
                batch_size=encode_kwargs["batch_size"],
                shuffle=False,
            ),
            task_name=self.task_name,
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
