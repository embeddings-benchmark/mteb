from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from datasets import Dataset, DatasetDict
from sklearn.metrics.pairwise import euclidean_distances
from torch.utils.data import DataLoader

import mteb
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ScoringFunction
from mteb.tasks.sts.dan.stakeholder_grounding import (
    StakeholderGrounding,
    _alpha_auc,
    _arrangement_triplets,
)
from mteb.types import Array, BatchedInput

THRESHOLDS = StakeholderGrounding.thresholds

# Positions of each text on a shared reference canvas
CANVAS = {
    "Children need more teachers": (0.0, 0.0),
    "Schools lack funding": (1.0, 0.0),
    "Teachers are overworked": (3.0, 1.0),
    "Hospitals have long waiting lists": (0.0, 4.0),
    "Elderly care is understaffed": (6.0, 6.0),
    "Nurses leave the profession": (2.0, 7.0),
}
POSITIONS = np.array(list(CANVAS.values()))
# A layout that agrees with some, but not all, of the triplets on the canvas: every text is moved to the position of
# the next text
DISTORTED_POSITIONS = np.roll(POSITIONS, -1, axis=0)


def _score(
    human: list[np.ndarray], model: list[np.ndarray], inverted: bool = False
) -> float:
    """alpha AUC of arrangements with human positions against a model embedding the texts at the given positions."""
    ratios, agreements = [], []
    for human_positions, model_positions in zip(human, model, strict=True):
        similarities = -euclidean_distances(model_positions)
        ratio, agree = _arrangement_triplets(
            euclidean_distances(human_positions),
            -similarities if inverted else similarities,
        )
        ratios.append(ratio)
        agreements.append(agree)
    return _alpha_auc(np.concatenate(ratios), np.concatenate(agreements), THRESHOLDS)


def test_model_reproducing_the_expert_layouts_scores_one():
    """A model whose similarities order every triplet as the experts did is in perfect agreement."""
    assert _score([POSITIONS], [POSITIONS]) == pytest.approx(1.0)


def test_model_inverting_the_expert_layouts_scores_minus_one():
    """A model that finds the items the experts placed far apart most similar is in perfect disagreement."""
    assert _score([POSITIONS], [POSITIONS], inverted=True) == pytest.approx(-1.0)


def test_score_does_not_depend_on_canvas_scale_or_position():
    """Only the relative placement of the texts matters, not where on the canvas or how spread out they are."""
    original_score = _score([POSITIONS], [DISTORTED_POSITIONS])
    moved_score = _score([3 * POSITIONS + [-1000, 42]], [DISTORTED_POSITIONS])

    assert -1 < original_score < 1
    assert moved_score == pytest.approx(original_score)


def test_score_does_not_depend_on_order_of_arrangements():
    """The arrangements are pooled, so the order in which they appear does not matter."""
    human = [POSITIONS[:5], 2 * POSITIONS[1:]]
    model = [DISTORTED_POSITIONS[:5], DISTORTED_POSITIONS[1:]]
    original_score = _score(human, model)
    reordered_score = _score(human[::-1], model[::-1])

    assert -1 < original_score < 1
    assert reordered_score == pytest.approx(original_score)


class _LayoutEncoder(AbsEncoder):
    """Embeds each text at a fixed 2D position; texts placed closer together are more similar."""

    def __init__(self, layout: dict[str, tuple[float, float]]) -> None:
        self.layout = layout
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(similarity_fn_name=ScoringFunction.CUSTOM)
        )

    def encode(self, inputs: DataLoader[BatchedInput], **kwargs: Any) -> Array:
        return np.array(
            [self.layout[text] for batch in inputs for text in batch["text"]]
        )

    def similarity(self, embeddings1: Array, embeddings2: Array) -> Array:
        return -euclidean_distances(embeddings1, embeddings2)


class _InMemoryStakeholderGrounding(StakeholderGrounding):
    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        texts = list(CANVAS)
        arrangements = [
            {
                "round_id": round_id,
                "texts": round_texts,
                "x": [CANVAS[text][0] * scale for text in round_texts],
                "y": [CANVAS[text][1] * scale for text in round_texts],
            }
            for round_id, round_texts, scale in [
                ("1-A", texts[:5], 100),
                ("1-B", texts[1:], 50),
            ]
        ]
        self.dataset = {"rai": DatasetDict({"test": Dataset.from_list(arrangements)})}
        self.dataset_transform()
        self.data_loaded = True


def test_evaluate_matches_helpers():
    """End to end, the pairs are grouped back into arrangements and scored like the helpers do."""
    task = _InMemoryStakeholderGrounding()
    task.load_data()
    assert task.dataset["rai"]["test"].num_rows == 2 * 10

    layout = dict(zip(CANVAS, DISTORTED_POSITIONS.tolist(), strict=True))
    results = mteb.evaluate(_LayoutEncoder(layout), task, cache=None, co2_tracker=False)

    expected = _score(
        [POSITIONS[:5], POSITIONS[1:]],
        [DISTORTED_POSITIONS[:5], DISTORTED_POSITIONS[1:]],
    )
    assert results[0].get_score() == pytest.approx(expected)
