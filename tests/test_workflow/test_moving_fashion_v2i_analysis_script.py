from __future__ import annotations

import json

import pytest

from scripts.analyze_moving_fashion_v2i_pair_predictions import analyze_predictions


def test_analyzes_predictions_by_source_and_is_deterministic(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.json"
    pairs_path = tmp_path / "pairs.json"
    predictions = {
        "default": {
            "test": {
                "similarity_scores": [0.9, 0.1, 0.8, 0.2],
                "cosine_scores": [0.9, 0.1, 0.8, 0.2],
                "manhattan_distances": [0.1, 0.9, 0.2, 0.8],
                "euclidean_distances": [0.1, 0.9, 0.2, 0.8],
                "dot_scores": [0.9, 0.1, 0.8, 0.2],
            }
        }
    }
    pairs = {
        "task_name": "MovingFashionV2IPairClassification",
        "dataset_revision": "test-revision",
        "rows": {
            "video_id": ["v1", "v1", "v2", "v2"],
            "image_id": ["i1", "i2", "i3", "i4"],
            "label": [1, 0, 1, 0],
            "source_subset": ["hard", "hard", "regular", "regular"],
        },
    }
    predictions_path.write_text(json.dumps(predictions))
    pairs_path.write_text(json.dumps(pairs))

    first = analyze_predictions(
        predictions_path,
        pairs_path,
        random_trials=20,
        bootstrap_samples=20,
        seed=7,
    )
    second = analyze_predictions(
        predictions_path,
        pairs_path,
        random_trials=20,
        bootstrap_samples=20,
        seed=7,
    )

    assert first == second
    assert first["overall"]["max_ap"] == pytest.approx(1.0)
    assert first["by_source"]["hard"]["num_pairs"] == 2
    assert first["by_source"]["regular"]["num_pairs"] == 2
    assert first["by_source"]["hard"]["metrics"]["max_ap"] == pytest.approx(1.0)
    assert first["video_cluster_bootstrap"]["point_estimate"] == pytest.approx(1.0)
    assert first["random_ranking"]["trials"] == 20


def test_rejects_prediction_and_pair_length_mismatch(tmp_path) -> None:
    predictions_path = tmp_path / "predictions.json"
    pairs_path = tmp_path / "pairs.json"
    scores = {
        "similarity_scores": [0.9],
        "cosine_scores": [0.9],
        "manhattan_distances": [0.1],
        "euclidean_distances": [0.1],
        "dot_scores": [0.9],
    }
    predictions_path.write_text(json.dumps(scores))
    pairs_path.write_text(
        json.dumps(
            {
                "task_name": "MovingFashionV2IPairClassification",
                "dataset_revision": "test-revision",
                "rows": {
                    "video_id": ["v1", "v1"],
                    "image_id": ["i1", "i2"],
                    "label": [1, 0],
                    "source_subset": ["hard", "hard"],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="lengths differ"):
        analyze_predictions(
            predictions_path,
            pairs_path,
            random_trials=1,
            bootstrap_samples=1,
        )
