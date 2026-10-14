from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import DataLoader

import mteb
from mteb.mocks.mock_tasks import MockSpatialArrangementTask
from mteb.mocks.mock_tasks.spatial_arrangement import MOCK_CANVAS
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ScoringFunction
from mteb.types import Array, BatchedInput

# A layout that agrees with some, but not all, of the expert judgements in the mock task: every text is moved to the
# position of the next text on the mock canvas
_positions = list(MOCK_CANVAS.values())
DISTORTED_CANVAS = dict(zip(MOCK_CANVAS, _positions[1:] + _positions[:1], strict=True))


class _LayoutEncoder(AbsEncoder):
    """Embeds each text at a fixed 2D position; texts placed closer together are more similar."""

    def __init__(
        self, layout: dict[str, tuple[float, float]], inverted: bool = False
    ) -> None:
        self.layout = layout
        self.inverted = inverted
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(similarity_fn_name=ScoringFunction.CUSTOM)
        )

    def encode(self, inputs: DataLoader[BatchedInput], **kwargs: Any) -> Array:
        return np.array(
            [self.layout[text] for batch in inputs for text in batch["text"]]
        )

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        distances = euclidean_distances(embeddings1, embeddings2)
        return distances if self.inverted else -distances


def _evaluate(model: AbsEncoder, task: MockSpatialArrangementTask) -> float:
    results = mteb.evaluate(model, task, cache=None, co2_tracker=False)
    return results[0].get_score()


def test_model_reproducing_the_expert_layouts_scores_one():
    """A model whose similarities order every triplet as the experts did is in perfect agreement."""
    score = _evaluate(_LayoutEncoder(MOCK_CANVAS), MockSpatialArrangementTask())

    assert score == pytest.approx(1.0)


def test_model_inverting_the_expert_layouts_scores_minus_one():
    """A model that finds the items the experts placed far apart most similar is in perfect disagreement."""
    score = _evaluate(
        _LayoutEncoder(MOCK_CANVAS, inverted=True), MockSpatialArrangementTask()
    )

    assert score == pytest.approx(-1.0)


def test_score_does_not_depend_on_canvas_scale_or_position():
    """Only the relative placement of the texts matters, not where on the canvas or how spread out they are."""
    model = _LayoutEncoder(DISTORTED_CANVAS)
    original_score = _evaluate(model, MockSpatialArrangementTask())

    task = MockSpatialArrangementTask()
    task.load_data()
    task.dataset["test"] = task.dataset["test"].map(
        lambda row: {
            "x": [3 * x - 1000 for x in row["x"]],
            "y": [3 * y + 42 for y in row["y"]],
        }
    )
    moved_score = _evaluate(model, task)

    assert -1 < original_score < 1
    assert moved_score == pytest.approx(original_score)


def test_score_does_not_depend_on_order_of_arrangements():
    """The arrangements are pooled, so the order in which they appear in the dataset does not matter."""
    model = _LayoutEncoder(DISTORTED_CANVAS)
    original_score = _evaluate(model, MockSpatialArrangementTask())

    task = MockSpatialArrangementTask()
    task.load_data()
    task.dataset["test"] = task.dataset["test"].select([1, 0])
    reordered_score = _evaluate(model, task)

    assert -1 < original_score < 1
    assert reordered_score == pytest.approx(original_score)
