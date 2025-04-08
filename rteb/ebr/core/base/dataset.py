from __future__ import annotations

from abc import ABC
from functools import cache
from pathlib import Path
from typing import Any, TYPE_CHECKING

from torch.utils.data import Dataset

if TYPE_CHECKING:
    from ebr.core.meta import DatasetMeta


def add_instruct(
    dataset: Dataset,
    instruct: str,
    input_type: str
):

    for item in dataset.data:
        if instruct:
            item["text"] = instruct + item["text"]
        item["input_type"] = input_type

    return dataset


class RetrievalDataset(ABC):

    LEADERBOARD: str = None

    def __init__(
        self,
        data_path: str,
        dataset_meta: DatasetMeta,
        query_instruct: str | None = None,
        corpus_instruct: str | None = None,
        **kwargs
    ):
        assert type(self).LEADERBOARD, f"leaderboard must be defined"
        super().__init__()
        self._dataset_meta = dataset_meta
        self._query_instruct = query_instruct
        self._corpus_instruct = corpus_instruct
        self._task_path = (Path(data_path) / dataset_meta.dataset_name).resolve()

    #def __getattr__(self, name: str) -> Any:
    #    try:
    #        return super().__getattr__(name)
    #    except AttributeError:
    #        return getattr(self._dataset_meta, name)

    @property
    @cache
    def corpus(self) -> Dataset:
        corpus = self._corpus()
        corpus = add_instruct(corpus, self._corpus_instruct, "document")
        return corpus

    def _corpus(self) -> Dataset:
        raise NotImplementedError

    @property
    @cache
    def queries(self) -> Dataset:
        queries = self._queries()
        queries = add_instruct(queries, self._query_instruct, "query")
        return queries

    def _queries(self) -> Dataset:
        raise NotImplementedError

    @property
    @cache
    def relevance(self) -> dict:
        # Dict of dict: relevance[query_id][corpus_id] = score
        pass

    def prepare_data(self):
        _ = self.corpus
        _ = self.queries
        _ = self.relevance