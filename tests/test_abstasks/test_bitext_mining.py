from __future__ import annotations

import pytest

import mteb
from mteb.abstasks.text.bitext_mining import AbsTaskBitextMining
from mteb.mocks.mock_tasks import (
    MockMultilingualBitextMiningTask,
    MockMultilingualParallelBitextMiningTask,
)


@pytest.mark.parametrize(
    "task",
    [MockMultilingualBitextMiningTask(), MockMultilingualParallelBitextMiningTask()],
    ids=["subsets", "parallel_subsets"],
)
def test_evaluate_only_runs_subsets_to_run(task: AbsTaskBitextMining):
    """`mteb.evaluate` passes the missing subsets as `subsets_to_run` when it resumes from the cache."""
    model = mteb.get_model("mteb/baseline-random-encoder")
    subsets_to_run = task.hf_subsets[-1:]

    scores = task.evaluate(
        model, "test", subsets_to_run=subsets_to_run, encode_kwargs={}
    )

    assert list(scores) == subsets_to_run
