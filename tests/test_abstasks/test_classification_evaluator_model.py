from __future__ import annotations

import pytest

import mteb
from mteb._evaluators.sklearn_evaluator import SklearnEvaluator
from mteb.abstasks import AbsTaskClassification


@pytest.mark.parametrize(
    "task",
    [
        task
        for task in mteb.get_tasks(
            exclude_superseded=False, exclude_aggregate=False, exclude_beta=False
        )
        if isinstance(task, AbsTaskClassification)
    ],
    ids=lambda task: task.metadata.name,
)
def test_classifier_is_set_on_evaluator_model(task: AbsTaskClassification) -> None:
    """A task that wants a non-default classifier must set `evaluator_model`.

    `evaluator` holds the evaluator class, so putting a classifier there leaves
    `evaluator_model` on its default and the classifier is never used.
    """
    assert isinstance(task.evaluator, type) and issubclass(
        task.evaluator, SklearnEvaluator
    ), (
        f"{task.metadata.name} sets `evaluator` to {task.evaluator!r}; "
        "a classifier belongs on `evaluator_model`"
    )
