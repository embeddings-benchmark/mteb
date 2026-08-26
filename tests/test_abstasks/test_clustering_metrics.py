from __future__ import annotations

import random

import numpy as np

import mteb
from mteb.abstasks.clustering import _evaluate_clustering_bootstrapped
from mteb.mocks.mock_tasks import LegacyMockClusteringFastTask
from tests.mock_models import MockSentenceTransformer


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


def test_random_labels_ami_near_zero():
    # AMI corrects for chance, so random labels vs k-means partition should
    # score near 0; v_measure does not have this guarantee.
    embeddings, _ = _well_separated_dataset(n_clusters=5, points_per_cluster=40, seed=1)
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


def test_end_to_end_result_dict_contains_both_metrics():
    results = mteb.evaluate(
        MockSentenceTransformer(),
        LegacyMockClusteringFastTask(),
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


def _ragged_dataset(dim: int = 8, seed: int = 0):
    """Two gold classes at level 1, plus documents whose label stops at level 0.

    The documents that stop at level 0 are placed on top of the two level-1
    classes, so they cannot be recovered as a group of their own. If they are
    scored at level 1 they drag the score down; if they are excluded, the two
    level-1 classes are perfectly separable.
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=10.0, size=(2, dim))
    embeddings: list[np.ndarray] = []
    labels: list[list[str]] = []
    for cluster_idx, center in enumerate(centers):
        for _ in range(12):
            embeddings.append(center + rng.normal(scale=0.01, size=dim))
            labels.append(["Top", f"Sub{cluster_idx}"])
    # Documents with no level-1 label, sitting on the same two centers.
    for center in centers:
        for _ in range(6):
            embeddings.append(center + rng.normal(scale=0.01, size=dim))
            labels.append(["Top"])
    return np.asarray(embeddings), labels


def _run(embeddings, labels):
    return _evaluate_clustering_bootstrapped(
        embeddings,
        labels,
        n_clusters=2,
        cluster_size=len(embeddings),
        kmean_batch_size=64,
        max_depth=None,
        rng_state=random.Random(0),
        seed=0,
    )


def test_documents_without_a_label_at_this_level_are_excluded():
    # The two level-1 classes are perfectly separable once the documents whose
    # label stops at level 0 are excluded, so v_measure at level 1 is 1.0.
    embeddings, labels = _ragged_dataset()
    scores, _ = _run(embeddings, labels)

    assert min(scores["v_measure"]["Level 1"]) == 1.0


def test_level_cluster_count_excludes_documents_without_a_label():
    # Two gold classes reach level 1, so k is 2. Counting the excluded
    # documents as a gold class of their own would make it 3.
    embeddings, labels = _ragged_dataset()
    _, assignments = _run(embeddings, labels)

    for assignment in assignments["Level 1"]:
        assert len(set(assignment)) == 2
    # Level 0 is reached by every document, so it is unaffected.
    for assignment in assignments["Level 0"]:
        assert len(set(assignment)) == 1
