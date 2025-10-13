"""test that the mteb.MTEB works as intended and that encoders are correctly called and passed the correct arguments."""

from __future__ import annotations

import logging

import pytest

import mteb
import mteb.overview
from tests.mock_models import (
    MockCLIPEncoder,
    MockMocoEncoder,
)
from tests.mock_tasks import (
    MockImageClusteringTask,
    MockImageTextPairClassificationTask,
    MockRetrievalTask,
)

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "task", [MockImageTextPairClassificationTask(), MockRetrievalTask()]
)
def test_task_modality_filtering(task, caplog):
    with caplog.at_level(logging.WARNING):
        model = MockMocoEncoder()
        results = mteb.evaluate(
            model,
            task,
            cache=None,
        )
        assert (
            f"Model {model.mteb_model_meta.name} support modalities {model.mteb_model_meta.modalities}"
            f" but the task {task.metadata.name} only supports {task.metadata.modalities}. Skipping task."
        ) in caplog.text
        assert len(results[0].scores) == 0


@pytest.mark.parametrize("task", [MockImageClusteringTask()])
def test_task_modality_filtering_model_modalities_more_than_task_modalities(task):
    scores = mteb.evaluate(
        MockCLIPEncoder(),
        task,
        cache=None,
    )
    assert len(scores) == 1
