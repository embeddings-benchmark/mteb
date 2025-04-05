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
                task_metadata,
                hf_split,
                hf_subset,
                task_name: str | None = None,
                batch_size=32,
                **kwargs,
            ) -> np.ndarray:
                return np.eye(len(sentences.dataset))

        model = Model()
        sentences = ["dog walked home", "cat walked home", "robot walked to the park"]
        labels = [1, 2, 3]
        dataset = Dataset.from_dict({"text": sentences, "labels": labels})
        clusterer = ClusteringEvaluator(
            dataset,
            task_metadata="",  # typing: ignore
            hf_subset="",
            hf_split="",
        )
        result = clusterer(model)

        assert result == {"v_measure": 1.0}
