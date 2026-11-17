from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import mteb
from mteb._evaluators.zeroshot_classification_evaluator import (
    ZeroShotClassificationEvaluator,
)
from mteb.abstasks import AbsTaskZeroShotClassification
from mteb.mocks.mock_tasks import (
    MockTextZeroShotClassificationTask,
    MockZeroShotClassificationTask,
)
from mteb.timing import TimingStack


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

    scores = results[0].scores["test"][0]
    assert scores["accuracy"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["f1_weighted"] == 1.0
    assert scores["precision"] == 1.0
    assert scores["precision_weighted"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["recall_weighted"] == 1.0
    assert scores["ap"] == 1.0
    assert scores["ap_weighted"] == 1.0


class _MultiVectorEncoder:
    """Returns one `(num_tokens, dim)` tensor per input, as late-interaction models do."""

    mteb_model_meta = SimpleNamespace(similarity_fn_name="MaxSim")

    def __init__(self) -> None:
        self.similarity_calls = 0

    def encode(self, inputs, **kwargs: Any):
        num_inputs = sum(len(next(iter(batch.values()))) for batch in inputs)
        return [torch.ones(3 + i, 4) for i in range(num_inputs)]

    def similarity(self, a, b):
        self.similarity_calls += 1
        return torch.tensor(
            [[float(i == j) for j in range(len(b))] for i in range(len(a))]
        )


def test_zeroshot_image_text_uses_the_models_similarity_for_multi_vector_models():
    """Image-text zero-shot used cosine similarity for every model, which fails on multi-vector
    embeddings; MaxSim models must be scored with their own `similarity`."""
    task = MockZeroShotClassificationTask()
    task.load_data()
    evaluator = ZeroShotClassificationEvaluator(
        task.dataset["test"],
        task.input_column_name,
        task.get_candidate_labels(),
        task_metadata=task.metadata,
        hf_split="test",
        hf_subset="default",
        timer=TimingStack(),
    )
    model = _MultiVectorEncoder()
    probs = evaluator(model, encode_kwargs={"batch_size": 2})
    assert model.similarity_calls == 1
    assert probs.argmax(1).tolist() == [0, 1]
