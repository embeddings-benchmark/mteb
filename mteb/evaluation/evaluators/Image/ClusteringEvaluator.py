from __future__ import annotations

import logging
from typing import Any

import sklearn
import sklearn.cluster
from datasets import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn import metrics

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import create_image_dataloader
from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)


class ImageClusteringEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        image_column_name: str,
        label_column_name: str,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        clustering_batch_size: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset = dataset
        self.image_column_name = image_column_name
        self.labels = dataset[label_column_name]
        self.clustering_batch_size = clustering_batch_size
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any]):
        image_embeddings = model.encode(
            create_image_dataloader(
                self.dataset,
                image_column_name=self.image_column_name,
                batch_size=encode_kwargs["batch_size"],
            ),
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
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
