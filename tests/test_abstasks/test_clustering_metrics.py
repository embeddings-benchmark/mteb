from __future__ import annotations

import random

import numpy as np

import mteb
from mteb.abstasks.clustering import _evaluate_clustering_bootstrapped
from tests.mock_models import MockSentenceTransformer
from tests.mock_tasks import MockClusteringFastTask


def _well_separated_dataset(
    n_clusters: int = 3, points_per_cluster: int = 12, dim: int = 8, seed: int = 0
):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=10.0, size=(n_clusters, dim))
    embeddings = []
    labels: list[list[str]] = []
    for cluster_idx, center in enumerate(centers):
        for _ in range(points_per_cluster):
            embeddings.append(center + rng.normal(scale=0.01, size=dim))
            labels.append([str(cluster_idx)])
    return np.asarray(embeddings), labels


class TestClusteringBootstrap:
    def test_returns_both_v_measure_and_ami(self):  # noqa: PLR6301
        embeddings, labels = _well_separated_dataset()
        scores, assignments = _evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=3,
            cluster_size=len(embeddings),
            kmean_batch_size=64,
            max_depth=None,
            rng_state=random.Random(0),
            seed=0,
        )

        assert set(scores) == {"v_measure", "ami"}
        assert set(scores["v_measure"]) == {"Level 0"}
        assert set(scores["ami"]) == {"Level 0"}
        assert len(scores["v_measure"]["Level 0"]) == 3
        assert len(scores["ami"]["Level 0"]) == 3
        assert set(assignments) == {"Level 0"}

    def test_perfect_clustering_scores_near_one(self):  # noqa: PLR6301
        embeddings, labels = _well_separated_dataset()
        scores, _ = _evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=2,
            cluster_size=len(embeddings),
            kmean_batch_size=64,
            max_depth=None,
            rng_state=random.Random(0),
            seed=0,
        )

        mean_v = float(np.mean(scores["v_measure"]["Level 0"]))
        mean_ami = float(np.mean(scores["ami"]["Level 0"]))
        assert mean_v > 0.99
        assert mean_ami > 0.99

    def test_random_labels_ami_near_zero(self):  # noqa: PLR6301
        # AMI corrects for chance, so random labels vs k-means partition should
        # score near 0; v_measure does not have this guarantee.
        embeddings, _ = _well_separated_dataset(
            n_clusters=5, points_per_cluster=40, seed=1
        )
        rng = random.Random(1)
        labels = [[str(rng.randrange(5))] for _ in range(len(embeddings))]
        scores, _ = _evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=4,
            cluster_size=len(embeddings),
            kmean_batch_size=64,
            max_depth=None,
            rng_state=random.Random(0),
            seed=0,
        )

        mean_ami = float(np.mean(scores["ami"]["Level 0"]))
        assert abs(mean_ami) < 0.1


class TestClusteringResultDict:
    def test_end_to_end_result_dict_contains_both_metrics(self):  # noqa: PLR6301
        results = mteb.evaluate(
            MockSentenceTransformer(),
            MockClusteringFastTask(),
            cache=None,
            co2_tracker=False,
        )

        scores = results[0].scores["test"][0]
        expected_keys = {
            "v_measure",
            "v_measure_std",
            "v_measures",
            "ami",
            "ami_std",
            "ami_scores",
        }
        assert expected_keys.issubset(scores.keys())
        assert isinstance(scores["v_measure"], float)
        assert isinstance(scores["ami"], float)
        assert isinstance(scores["v_measures"], dict)
        assert isinstance(scores["ami_scores"], dict)
