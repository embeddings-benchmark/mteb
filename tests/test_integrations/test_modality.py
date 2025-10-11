"""test that the mteb.MTEB works as intended and that encoders are correctly called and passed the correct arguments."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

import mteb
import mteb.overview
from mteb.MTEB import logger
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


# TODO: KCE we might need to add support for this in mteb.evaluate
# NOTE: Covers image and image-text tasks. Can be extended to cover new mixed-modality task types.
@pytest.mark.parametrize(
    "task", [MockImageTextPairClassificationTask(), MockRetrievalTask()]
)
@patch.object(logger, "info")
def test_task_modality_filtering(mock_logger, task):
    eval = mteb.MTEB(tasks=[task])

    # Run the evaluation
    eval.run(
        model=MockMocoEncoder(),
        output_folder="tests/results",
        overwrite_results=True,
    )

    # Check that the task was skipped and the correct log message was generated
    task_modalities = ", ".join(
        f"'{modality}'" for modality in sorted(task.metadata.modalities)
    )
    mock_logger.assert_called_with(
        f"mock/MockMocoModel only supports ['image'], but the task modalities are [{task_modalities}]."
    )


# TODO: KCE: Unsure if we need this test:
@pytest.mark.parametrize("task", [MockImageClusteringTask()])
def test_task_modality_filtering_model_modalities_more_than_task_modalities(task):
    eval = mteb.MTEB(tasks=[task])

    # Run the evaluation
    eval.run(
        model=MockCLIPEncoder(),
        output_folder="tests/results",
        overwrite_results=True,
    )
