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
def test_cross_encoder_on_task(task: AbsTask):
    """Ensures that cross-encoders can be run on retrieval and reranking tasks"""
    model = mteb.get_model("baseline/random-cross-encoder-baseline")
    mteb.evaluate(model, task, cache=None)
