import logging
from typing import Any

from datasets import Dataset
from scipy.optimize import linear_sum_assignment
from sklearn import cluster, metrics

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import create_dataloader
from mteb.models import Encoder

from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class ClusteringEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        input_column_name: str,
        label_column_name: str,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        clustering_batch_size: int = 500,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset
        self.clustering_batch_size = clustering_batch_size
        self.input_column_name = input_column_name
        self.label_column_name = label_column_name
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
        v_measure_only: bool = False,
    ) -> dict[str, float]:
        data_loader = create_dataloader(
            self.dataset,
            self.task_metadata,
            input_column=self.input_column_name,
            batch_size=encode_kwargs["batch_size"],
        )

        logger.info("Running clustering - Encoding samples...")
        embeddings = model.encode(
            data_loader,
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        labels = self.dataset[self.label_column_name]

        logger.info("Running clustering - Fitting Mini-Batch K-Means...")
        clustering_model = cluster.MiniBatchKMeans(
            n_clusters=len(set(labels)),
            batch_size=self.clustering_batch_size,
            n_init="auto",
        )
        clustering_model.fit(embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Running clustering - Evaluating clustering...")
        v_measure = metrics.cluster.v_measure_score(labels, cluster_assignment)
        if v_measure_only:
            return {"v_measure": float(v_measure)}

        nmi = metrics.cluster.normalized_mutual_info_score(labels, cluster_assignment)
        ari = metrics.cluster.adjusted_rand_score(labels, cluster_assignment)

        matrix = metrics.confusion_matrix(labels, cluster_assignment)
        # get linear sum assignment
        row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
        total_correct = matrix[row_ind, col_ind].sum()
        clustering_accuracy = total_correct / len(labels)

        return {
            "v_measure": float(v_measure),
            "nmi": float(nmi),
            "ari": float(ari),
            "cluster_accuracy": float(clustering_accuracy),
        }
