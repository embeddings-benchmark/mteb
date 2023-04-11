from typing import List

import numpy as np

from mteb.evaluation.evaluators import ClusteringEvaluator


class TestClusteringEvaluator:
    def test_clustering_v_measure(self):
        class Model:
            def encode(self, sentences: List[str]) -> np.ndarray:
                return np.eye(len(sentences))

        model = Model()
        sentences = ["dog walked home", "cat walked home", "robot walked to the park"]
        clusterer = ClusteringEvaluator(sentences=sentences, labels=[1, 2, 3])
        result = clusterer(model)

        assert result == {"v_measure": 1.0}
