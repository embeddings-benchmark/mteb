from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsClassifier

import mteb
from mteb.abstasks.multilabel_classification import _evaluate_classifier
from mteb.mocks.mock_tasks import MockMultilabelClassification

if TYPE_CHECKING:
    from pathlib import Path


class _RenamedLabelMultilabelClassification(MockMultilabelClassification):
    label_column_name = "labels"

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        super().load_data(num_proc=num_proc, **kwargs)
        self.dataset = self.dataset.rename_column("label", "labels")


class _ManyNeighborsMultilabelClassification(MockMultilabelClassification):
    # the mock train split only has 6 rows
    evaluator_model = KNeighborsClassifier(n_neighbors=50)


@pytest.mark.parametrize(
    ("task_cls", "max_eval_samples", "expected_rows"),
    [
        (MockMultilabelClassification, 4, 4),
        (MockMultilabelClassification, 6, 6),
        (MockMultilabelClassification, None, 6),
        (_RenamedLabelMultilabelClassification, 4, 4),
    ],
)
def test_max_eval_samples_caps_the_scored_rows(
    tmp_path: Path,
    task_cls: type[MockMultilabelClassification],
    max_eval_samples: int | None,
    expected_rows: int,
):
    task = task_cls()
    task.max_eval_samples = max_eval_samples
    model = mteb.get_model_meta("mteb/baseline-random-encoder")
    mteb.evaluate(model, task, prediction_folder=tmp_path, cache=None)

    with task._predictions_path(tmp_path).open() as f:
        predictions = json.load(f)["default"]["test"]

    assert len(predictions) == task.n_experiments
    assert all(len(experiment) == expected_rows for experiment in predictions)


def test_evaluate_classifier_caps_n_neighbors_to_the_train_size():
    rng = np.random.default_rng(42)
    classifier = KNeighborsClassifier(n_neighbors=5)
    y_pred, fitted = _evaluate_classifier(
        rng.normal(size=(3, 4)),
        np.array([[1, 0], [0, 1], [1, 1]]),
        rng.normal(size=(7, 4)),
        classifier,
    )

    assert y_pred.shape == (7, 2)
    assert fitted.n_neighbors == 3
    assert classifier.n_neighbors == 5  # the configured classifier is left alone


def test_task_with_fewer_train_samples_than_n_neighbors_can_be_evaluated(
    tmp_path: Path,
):
    task = _ManyNeighborsMultilabelClassification()
    model = mteb.get_model_meta("mteb/baseline-random-encoder")
    mteb.evaluate(model, task, prediction_folder=tmp_path, cache=None, raise_error=True)

    with task._predictions_path(tmp_path).open() as f:
        predictions = json.load(f)["default"]["test"]

    assert len(predictions) == task.n_experiments
