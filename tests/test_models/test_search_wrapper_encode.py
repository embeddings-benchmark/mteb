import pytest

import mteb
from mteb import AbsTask, EncoderProtocol
from mteb.models import SearchEncoderWrapper
from tests.mock_tasks import (
    MockBitextMiningTask,
    MockPairClassificationTask,
    MockRetrievalTask,
)


@pytest.mark.parametrize(
    "task",
    [
        MockRetrievalTask(),
        MockPairClassificationTask(),  # uses model.similarity_pairwise
        MockBitextMiningTask(),  # uses model.similarity
    ],
)
@pytest.mark.parametrize("model", [mteb.get_model("baseline/random-encoder-baseline")])
def test_benchmark_datasets(task: AbsTask, model: mteb.EncoderProtocol):
    """Test that a task can be fetched and run"""
    model = SearchEncoderWrapper(model)
    assert isinstance(model, EncoderProtocol)
    mteb.evaluate(model, task, cache=None)
