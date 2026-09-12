from __future__ import annotations

from typing import Any

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb._evaluators import BitextMiningEvaluator
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.mocks.mock_tasks.bitext_mining import MockBitextMiningTask
from mteb.timing import TimingStack


class DummySimilarityModel:
    def similarity(
        self, a: torch.Tensor, b: torch.Tensor
    ) -> list[list[float]] | torch.Tensor:
        a_norm = a / a.norm(dim=-1, keepdim=True)
        b_norm = b / b.norm(dim=-1, keepdim=True)
        return a_norm @ b_norm.T

    def encode(
        self,
        dataloader: DataLoader,
        *,
        task_metadata: TaskMetadata | None = None,
        hf_subset: str | None = None,
        hf_split: str | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        n = len(dataloader.dataset)
        return torch.randn(n, 8)


@pytest.fixture
def evaluator() -> BitextMiningEvaluator:
    dataset = Dataset.from_dict({"sentence1": ["hello"], "sentence2": ["world"]})
    return BitextMiningEvaluator(
        sentences=dataset,
        task_metadata=MockBitextMiningTask.metadata,  # type: ignore[arg-type]
        hf_split="test",
        hf_subset="default",
        pair_columns=[("sentence1", "sentence2")],
        timer=TimingStack(),
    )


@pytest.mark.parametrize(
    ("query_shape", "corpus_shape"),
    [
        ((4,), (4,)),  # 1D query, 1D corpus (exercises corpus unpacking fix)
        ((2, 4), (4,)),  # 2D query, 1D corpus
        ((4,), (3, 4)),  # 1D query, 2D corpus
        ((2, 4), (3, 4)),  # 2D query, 2D corpus
    ],
)
def test_similarity_search_shapes(
    evaluator: BitextMiningEvaluator,
    query_shape: tuple[int, ...],
    corpus_shape: tuple[int, ...],
) -> None:
    model = DummySimilarityModel()
    queries = torch.randn(*query_shape)
    corpus = torch.randn(*corpus_shape)

    results = evaluator._similarity_search(queries, corpus, model=model)  # type: ignore[arg-type]

    expected_len = 1 if len(query_shape) == 1 else query_shape[0]
    assert len(results) == expected_len
    for res in results:
        assert isinstance(res, dict)
        assert "corpus_id" in res
        assert "score" in res
        assert isinstance(res["score"], float)


def test_bitext_mining_evaluator_call(evaluator: BitextMiningEvaluator) -> None:
    model = DummySimilarityModel()
    results = evaluator(model, encode_kwargs={"batch_size": 1})  # type: ignore[arg-type]
    assert "sentence1-sentence2" in results
    assert len(results["sentence1-sentence2"]) == 1
    assert "corpus_id" in results["sentence1-sentence2"][0]
    assert "score" in results["sentence1-sentence2"][0]
