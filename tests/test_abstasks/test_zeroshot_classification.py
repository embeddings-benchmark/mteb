from __future__ import annotations

import pytest

import mteb
from mteb.abstasks import AbsTaskZeroShotClassification
from mteb.mocks.mock_tasks import MockTextZeroShotClassificationTask


def test_normalize_labels_keeps_integer_labels():
    assert AbsTaskZeroShotClassification._normalize_labels([1, 0], ["a", "b"]) == [
        1,
        0,
    ]


def test_normalize_labels_maps_string_labels_to_candidate_indices():
    assert AbsTaskZeroShotClassification._normalize_labels(
        ["b", "a", "b"], ["a", "b"]
    ) == [1, 0, 1]


def test_normalize_labels_raises_on_unknown_string_labels():
    with pytest.raises(ValueError, match="get_candidate_labels"):
        AbsTaskZeroShotClassification._normalize_labels(["c", "a"], ["a", "b"])


def test_zeroshot_classification_scores_with_string_labels():
    """String labels are scored correctly with scikit-learn >= 1.9.

    Regression test for https://github.com/embeddings-benchmark/mteb/issues/4784:
    predictions are integer indices into the candidate labels, and mixing them
    with string labels raises an error in scikit-learn >= 1.9 (and silently
    produced an accuracy of 0.0 before).
    """
    model = mteb.get_model_meta("mteb/baseline-random-encoder")
    results = mteb.evaluate(
        model, MockTextZeroShotClassificationTask(), cache=None, co2_tracker=False
    )

    assert results[0].scores["test"][0]["accuracy"] == 1.0
