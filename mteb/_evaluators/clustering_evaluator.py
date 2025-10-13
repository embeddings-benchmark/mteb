import logging
from typing import Any

from datasets import Dataset
from sklearn import cluster

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
    ) -> list[int]:
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
            compute_labels=True,
            random_state=self.seed,
        )
        clustering_model.fit(embeddings)
        return clustering_model.labels_.tolist()
