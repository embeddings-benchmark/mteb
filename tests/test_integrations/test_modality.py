"""test that the mteb.MTEB works as intended and that encoders are correctly called and passed the correct arguments."""

from __future__ import annotations

import logging
from copy import deepcopy

import pytest

import mteb
from mteb.types import PromptType
from tests.mock_tasks import (
    MockImageClusteringTask,
    MockImageTextPairClassificationTask,
    MockMultiChoiceTask,
    MockPairClassificationTask,
    MockRetrievalTask,
)

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "task, modalities",
    [
        # Task needs image and text, model only text
        (MockImageTextPairClassificationTask(), ["text"]),
        # Retrieval task needs text, model only image
        (MockRetrievalTask(), ["image"]),
        # Task needs text, model only image
        (MockPairClassificationTask(), ["image"]),
        # Task needs text and images for query and image passage, model only text
        (MockMultiChoiceTask(), ["text"]),
        # First task needs text only, second image only, model has only text
        ((MockRetrievalTask(), MockImageClusteringTask()), ["text"]),
    ],
)
def test_task_modality_filtering(task, modalities):
    model_name = "baseline/random-encoder-baseline"
    model = mteb.get_model(model_name)
    model_meta = deepcopy(model.mteb_model_meta)
    model_meta.modalities = modalities
    model.mteb_model_meta = model_meta

    with pytest.raises(ValueError):
        mteb.evaluate(
            model,
            task,
            cache=None,
        )


@pytest.mark.parametrize("task", [MockMultiChoiceTask()])
def test_task_modality_filtering_model_modalities_only_one_of_modalities(task, caplog):
    """Task have it2i, model only image."""
    with caplog.at_level(logging.WARNING):
        model = mteb.get_model("baseline/random-encoder-baseline")
        model_meta = deepcopy(model.mteb_model_meta)
        model_meta.modalities = ["image"]
        model.mteb_model_meta = model_meta
        scores = mteb.evaluate(
            model,
            task,
            cache=None,
        )
        assert (
            f"Model {model.mteb_model_meta.name} supports {model.mteb_model_meta.modalities}, partially overlapping with"
            f" task {task.metadata.name} query={task.metadata.get_modalities(PromptType.query)},"
            f" document={task.metadata.get_modalities(PromptType.document)}. Performance might be suboptimal."
        ) in caplog.text
    assert len(scores) == 1


@pytest.mark.parametrize("task", [MockImageClusteringTask()])
def test_task_modality_filtering_model_modalities_more_than_task_modalities(task):
    scores = mteb.evaluate(
        mteb.get_model("baseline/random-encoder-baseline"),
        task,
        cache=None,
    )
    assert len(scores) == 1
