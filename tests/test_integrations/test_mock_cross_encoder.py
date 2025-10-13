import pytest

import mteb
from mteb.abstasks import AbsTask
from tests.mock_tasks import (
    MockInstructionReranking,
    MockMultilingualInstructionReranking,
    MockMultilingualRerankingTask,
    MockRerankingTask,
)


@pytest.mark.parametrize(
    "task",
    [
        MockRerankingTask(),
        MockMultilingualRerankingTask(),
        MockInstructionReranking(),
        MockMultilingualInstructionReranking(),
    ],
)
def test_mock_cross_encoder(task: AbsTask):
    """Test that a task can be fetched and run"""
    model = mteb.get_model("mteb/random-crossencoder-baseline")
    mteb.evaluate(model, task, cache=None)
