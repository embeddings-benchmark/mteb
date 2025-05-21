from __future__ import annotations

import pytest

import mteb
from mteb.abstasks.AbsTask import AbsTask
from mteb.encoder_interface import Encoder
from tests.test_benchmark.mock_models import MockSentenceTransformer
from tests.test_benchmark.mock_tasks import MockRetrievalTask


@pytest.mark.parametrize(
    "model, task", [(MockSentenceTransformer(), MockRetrievalTask())]
)
def test_run_task(model: Encoder, task: AbsTask):
    results = mteb.run_tasks(model, task, cache=None)

    assert len(results) == 0
    result = results[0]

    assert result.task_name == task.metadata.name, "results should match the task"
    assert set(result.eval_splits) == set(task.eval_splits), "splits should match task."
