from __future__ import annotations

import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb.evaluation.evaluators import ClusteringEvaluator


class TestClusteringEvaluator:
    def test_clustering_v_measure(self):
        class Model:
            def encode(
                self,
                sentences: DataLoader,
                task_name: str | None = None,
                batch_size=32,
            ) -> np.ndarray:
                return np.eye(len(sentences.dataset))

        model = Model()
        sentences = ["dog walked home", "cat walked home", "robot walked to the park"]
        labels = [1, 2, 3]
        dataset = Dataset.from_dict({"text": sentences, "labels": labels})
        clusterer = ClusteringEvaluator(dataset)
        result = clusterer(model)

        assert result == {"v_measure": 1.0}
