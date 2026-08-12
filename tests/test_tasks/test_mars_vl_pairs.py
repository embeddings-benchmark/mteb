from unittest.mock import patch

import pytest
from datasets import Dataset

from mteb.tasks.retrieval.eng.mars_vl_pairs import (
    MarsVLPairsI2TRetrieval,
    MarsVLPairsT2IRetrieval,
)


@pytest.mark.parametrize("task_cls", [MarsVLPairsT2IRetrieval, MarsVLPairsI2TRetrieval])
def test_mars_vl_pairs_uses_shared_pair_ids(task_cls):
    pairs = Dataset.from_dict(
        {
            "key": ["first", "second"],
            "refined_caption": ["First caption", "Second caption"],
            "image": [None, None],
        }
    )
    task = task_cls()

    with (
        patch("mteb.tasks.retrieval.eng.mars_vl_pairs._FROZEN_PAIRS", 2),
        patch(
            "mteb.tasks.retrieval.eng.mars_vl_pairs.load_dataset",
            return_value=pairs,
        ),
    ):
        task.load_data()

    split = task.dataset["default"]["test"]
    assert split["queries"]["id"] == ["first", "second"]
    assert split["corpus"]["id"] == ["first", "second"]
    assert split["relevant_docs"] == {
        "first": {"first": 1},
        "second": {"second": 1},
    }


@pytest.mark.parametrize("task_cls", [MarsVLPairsT2IRetrieval, MarsVLPairsI2TRetrieval])
def test_mars_vl_pairs_computes_full_gallery_mrr_separately(task_cls):
    task = task_cls()
    qrels = {"first": {"first": 1}, "second": {"second": 1}}
    results = {
        "first": {"first": 0.9, "second": 0.1},
        "second": {"first": 0.9, "second": 0.8},
    }

    scores = task.task_specific_scores({}, qrels, results, "test", "default")

    assert task.k_values == (1, 3, 5, 10, 20, 100, 1000)
    assert task._top_k == 2_247
    assert scores == {"mrr_at_2247": pytest.approx(0.75)}
