from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

import mteb
from mteb.abstasks import AbsTaskClassification, AbsTaskMultilabelClassification
from mteb.mocks.mock_tasks import MockMultilabelClassification

if TYPE_CHECKING:
    from pathlib import Path


class _RenamedLabelMultilabelClassification(MockMultilabelClassification):
    label_column_name = "labels"

    def load_data(self, num_proc: int | None = None, **kwargs: Any) -> None:
        super().load_data(num_proc=num_proc, **kwargs)
        self.dataset = self.dataset.rename_column("label", "labels")


def test_multilabel_tasks_set_their_classifier_on_evaluator_model():
    """Multilabel evaluation fits `evaluator_model` and never reads `evaluator`.

    `evaluator` is the evaluator class of single-label classification, so a classifier
    assigned to it is ignored and the task is scored with the default kNN instead.
    """
    tasks = mteb.get_tasks(exclude_superseded=False)
    overridden = [
        task.metadata.name
        for task in tasks
        if isinstance(task, AbsTaskMultilabelClassification)
        and task.evaluator is not AbsTaskClassification.evaluator
    ]
    assert overridden == []


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
