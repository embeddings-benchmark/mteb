"""test that mteb works with various output types of encoders"""

from __future__ import annotations

import logging

import pytest

import mteb
from mteb.abstasks import AbsTask
from tests.mock_models import (
    MockNumpyEncoder,
    MockSentenceTransformersbf16Encoder,
    MockTorchEncoder,
    MockTorchfp16Encoder,
)
from tests.mock_tasks import (
    MockBitextMiningTask,
    MockClassificationTask,
    MockClusteringFastTask,
    MockClusteringTask,
    MockInstructionReranking,
    MockInstructionRetrieval,
    MockMultilabelClassification,
    MockPairClassificationTask,
    MockRegressionTask,
    MockRerankingTask,
    MockRetrievalDialogTask,
    MockRetrievalTask,
    MockSTSTask,
    MockSummarizationTask,
    MockTextZeroShotClassificationTask,
)

logging.basicConfig(level=logging.INFO)


@pytest.mark.parametrize(
    "task",
    [
        MockBitextMiningTask(),
        MockClassificationTask(),
        MockRegressionTask(),
        MockClusteringTask(),
        MockClusteringFastTask(),
        MockPairClassificationTask(),
        MockRerankingTask(),
        MockRetrievalTask(),
        MockSTSTask(),
        MockMultilabelClassification(),
        MockSummarizationTask(),
        MockInstructionRetrieval(),
        MockInstructionReranking(),
        MockRetrievalDialogTask(),
        MockTextZeroShotClassificationTask(),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        MockNumpyEncoder(),
        MockTorchEncoder(),
        MockTorchfp16Encoder(),
        MockSentenceTransformersbf16Encoder(),
    ],
)
def test_encoder_dtype_on_task(task: AbsTask, model: mteb.Encoder):
    mteb.evaluate(model, task, cache=None)
