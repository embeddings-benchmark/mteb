from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from datasets import Dataset
from sklearn.metrics import v_measure_score
from torch.utils.data import DataLoader

from mteb._evaluators import ClusteringEvaluator
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks.clustering import MockClusteringTask
from mteb.timing import TimingStack

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestClusteringEvaluator:
    def test_clustering_v_measure(self):
        class Model:
            def encode(
                self,
                sentences: DataLoader,
                task_metadata: TaskMetadata,
                hf_split: str,
                hf_subset: str,
                task_name: str | None = None,
                batch_size: int = 32,
                **kwargs: Any,
            ) -> NDArray[np.floating]:
                return np.eye(len(sentences.dataset))

        model = Model()
        sentences = ["dog walked home", "cat walked home", "robot walked to the park"]
        labels = [1, 2, 3]
        dataset = Dataset.from_dict({"text": sentences, "labels": labels})

        clusterer = ClusteringEvaluator(
            dataset,
            input_column_name="text",
            label_column_name="labels",
            task_metadata=MockClusteringTask.metadata,  # typing: ignore
            hf_subset="",
            hf_split="",
            timer=TimingStack(),
        )
        result = clusterer(model, encode_kwargs={"batch_size": 32})

        assert v_measure_score(labels, result) == 1.0
