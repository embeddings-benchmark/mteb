from functools import cache
import json
import os

from torch.utils.data import Dataset

from ebr.core.base import RetrievalDataset
from ebr.core.meta import DatasetMeta
from ebr.utils.data import JSONLDataset


class TextRetrievalDataset(RetrievalDataset):

    LEADERBOARD: str = "Text"

    def __init__(
        self,
        data_path: str,
        dataset_meta: DatasetMeta,
        query_instruct: str | None = None,
        corpus_instruct: str | None = None,
        **kwargs
    ):
        super().__init__(
            data_path,
            dataset_meta,
            query_instruct=query_instruct,
            corpus_instruct=corpus_instruct,
            **kwargs
        )
        assert os.path.isdir(self._task_path), f"{self._task_path} is not a directory."

    @property
    def corpus_file(self) -> str:
        for name in ["corpus.jsonl", "corpus.arrow"]:
            file = os.path.join(self._task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Corpus file (corpus.{{jsonl/arrow}}) does not exist under {self._task_path}."
        )

    @cache
    def _corpus(self) -> Dataset:
        return JSONLDataset(self.corpus_file)

    @property
    def queries_file(self) -> str:
        for name in ["queries.jsonl", "queries.arrow"]:
            file = os.path.join(self._task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Queries file (queries.{{jsonl/arrow}}) does not exist under {self._task_path}."
        )

    @cache
    def _queries(self) -> Dataset:
        return JSONLDataset(self.queries_file)

    @property
    def relevance_file(self) -> str:
        for name in ["relevance.json", "relevance.jsonl"]:
            file = os.path.join(self._task_path, name)
            if os.path.exists(file):
                return file
        raise FileNotFoundError(
            f"Relevance file (relevance.{{json/jsonl}}) does not exist under {self._task_path}."
        )

    @property
    @cache
    def relevance(self) -> dict:
        relevant_docs = {}
        try:
            print(self.relevance_file)
            with open(self.relevance_file) as f:
                for line in f:
                    data = json.loads(line)
                    for key, value in data.items():
                        if key not in relevant_docs:
                            relevant_docs[key] = value
                        else:
                            relevant_docs[key].update(value)
        except FileNotFoundError:
            return {}
        return relevant_docs


# Legal datasets

AILACasedocs = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="AILACasedocs",
    tier=3,
    groups={"text": 1, "legal": 1, "english": 1},
    reference=None
)

AILAStatutes = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="AILAStatutes",
    tier=3,
    groups={"text": 1, "legal": 1, "english": 1},
    reference=None
)

LegalSummarization = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="LegalSummarization",
    tier=3,
    groups={"text": 1, "legal": 1, "english": 1},
    reference=None
)

LegalQuAD = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="LegalQuAD",
    tier=3,
    groups={"text": 1, "legal": 1, "german": 1},
    reference=None
)


# Finance datasets

FinanceBench = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="FinanceBench",
    tier=3,
    groups={"text": 1, "finance": 1, "english": 1},
    reference=None
)

HC3Finance = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="HC3Finance",
    tier=3,
    groups={"text": 1, "finance": 1, "english": 1},
    reference=None
)

FinQA = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="FinQA",
    tier=3,
    groups={"text": 1, "finance": 1, "english": 1},
    reference=None
)


# Code datasets

APPS = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="APPS",
    tier=3,
    groups={"text": 1, "code": 1, "english": 1},
    reference=None
)

DS1000 = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="DS1000",
    tier=3,
    groups={"text": 1, "code": 1, "english": 1},
    reference=None
)

HumanEval = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="HumanEval",
    tier=3,
    groups={"text": 1, "code": 1},
    reference=None
)

MBPP = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="MBPP",
    tier=3,
    groups={"text": 1, "code": 1},
    reference=None
)

WikiSQL = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="WikiSQL",
    tier=3,
    groups={"text": 1, "code": 1, "english": 1},
    reference=None
)


# Healthcare datasets

ChatDoctor_HealthCareMagic = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="ChatDoctor_HealthCareMagic",
    tier=3,
    groups={"text": 1, "healthcare": 1, "english": 1},
    reference=None
)


# Other/multilingual datasets

FrenchBoolQ = DatasetMeta(
    loader=TextRetrievalDataset,
    dataset_name="FrenchBoolQ",
    tier=3,
    groups={"text": 1, "french": 1},
    reference=None
)


