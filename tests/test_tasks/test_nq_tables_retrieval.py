"""Offline adapter tests. Fixtures are schema probes, not new benchmark data."""

from collections import Counter
from typing import Any
from unittest.mock import patch

import pytest
from datasets import Dataset

from mteb.get_tasks import _TASKS_REGISTRY
from mteb.tasks.retrieval.eng.nq_tables_retrieval import NQTablesRetrieval


@pytest.fixture
def release():
    corpus = Dataset.from_list(
        [
            {
                "_id": "Table_é_01",
                "title": "",
                "text": "Page\n\n| é |\n|---|\n| TBD |\n",
            },
            {"_id": "table_02", "title": "", "text": "  | x |\n|---|\n| 2 |  \n"},
        ]
    )
    source = {("corpus_md", "corpus_md"): corpus}
    for split in NQTablesRetrieval.supported_splits:
        qid = f"{split}_0001"
        source["queries", f"{split}_queries"] = Dataset.from_list(
            [
                {"_id": qid, "text": "  who? é\n"},
                {"_id": qid, "text": "  who? é\n"},
                {"_id": f"{split}_unjudged", "text": "Unjudged query"},
            ]
        )
        source["default", split] = Dataset.from_list(
            [
                {"qid": qid, "did": "Table_é_01", "score": 1},
                {"qid": qid, "did": "table_02", "score": 2},
            ]
        )
    return source


@pytest.fixture
def loader(release):
    def load(*, path, revision, name, split, **kwargs: Any):
        assert path == "ibm-research/NQTablesRetrieval"
        assert revision == "4962c33f5e651c82bc82c013893061202a47685e"
        return release[name, split]

    with patch(
        "mteb.tasks.retrieval.eng.nq_tables_retrieval.load_dataset", side_effect=load
    ) as mock:
        yield mock


def test_registered_complete_metadata():
    assert _TASKS_REGISTRY["NQTablesRetrieval"] is NQTablesRetrieval
    assert NQTablesRetrieval.metadata.license == "cc-by-4.0"
    NQTablesRetrieval.metadata._validate_metadata()
    assert NQTablesRetrieval.metadata.is_filled()
    assert list(NQTablesRetrieval.metadata.descriptive_stats or {}) == ["test"]


@pytest.mark.parametrize(
    "splits", [["train"], ["dev"], ["test"], ["train", "dev", "test"]]
)
def test_split_routing_and_shared_corpus(loader, splits):
    task = NQTablesRetrieval().filter_eval_splits(splits)
    task.load_data(cache_dir="local-cache", num_proc=1)
    assert set(task.dataset["default"]) == set(splits)
    corpora = [data["corpus"] for data in task.dataset["default"].values()]
    assert all(corpus is corpora[0] for corpus in corpora)
    calls = [
        (call.kwargs["name"], call.kwargs["split"]) for call in loader.call_args_list
    ]
    assert Counter(calls) == Counter(
        [("corpus_md", "corpus_md")]
        + [("queries", f"{split}_queries") for split in splits]
        + [("default", split) for split in splits]
    )
    assert all(
        call.kwargs["cache_dir"] == "local-cache" for call in loader.call_args_list
    )
    task.load_data()
    assert loader.call_count == 1 + 2 * len(splits)


def test_default_test_and_incremental_loading(loader):
    task = NQTablesRetrieval()
    task.load_data()
    assert list(task.dataset["default"]) == ["test"]
    corpus = task.dataset["default"]["test"]["corpus"]
    task.filter_eval_splits(["train", "dev", "test"]).load_data()
    assert len(task.dataset["default"]) == 3
    assert loader.call_count == 7
    assert task.dataset["default"]["train"]["corpus"] is corpus


def test_preserves_ids_text_rows_and_qrels(loader, release):
    task = NQTablesRetrieval()
    task.load_data()
    data = task.dataset["default"]["test"]
    assert (
        data["corpus"].rename_column("id", "_id").to_list()
        == release["corpus_md", "corpus_md"].to_list()
    )
    assert (
        data["queries"].rename_column("id", "_id").to_list()
        == release["queries", "test_queries"].to_list()
    )
    # Repeated identical queries and unjudged queries are retained, never filtered.
    assert data["queries"]["id"] == ["test_0001", "test_0001", "test_unjudged"]
    assert data["relevant_docs"] == {"test_0001": {"Table_é_01": 1, "table_02": 2}}
    assert task.source_qrels["test"].to_list() == release["default", "test"].to_list()
    assert data["top_ranked"] is None


@pytest.mark.parametrize(
    ("field", "value"), [("qid", "missing-query"), ("did", "missing-table")]
)
def test_preserves_dangling_qrels_without_repair(loader, release, field, value):
    rows = release["default", "test"].to_list()
    rows[0][field] = value
    release["default", "test"] = Dataset.from_list(rows)
    task = NQTablesRetrieval()
    task.load_data()
    assert task.source_qrels["test"].to_list() == rows
    rels = task.dataset["default"]["test"]["relevant_docs"]
    assert rels[rows[0]["qid"]][rows[0]["did"]] == rows[0]["score"]


@pytest.mark.parametrize("score", [1, 2])
def test_repeated_qrel_pair_is_not_silently_overwritten(loader, release, score):
    rows = release["default", "test"].to_list()
    rows.append({**rows[0], "score": score})
    release["default", "test"] = Dataset.from_list(rows)
    with pytest.raises(ValueError, match="Repeated qrel pair"):
        NQTablesRetrieval().load_data()


def test_conflicting_query_id_is_not_silently_overwritten(loader, release):
    rows = release["queries", "test_queries"].to_list()
    rows[1]["text"] = "conflicting text"
    release["queries", "test_queries"] = Dataset.from_list(rows)
    with pytest.raises(ValueError, match="Conflicting query text"):
        NQTablesRetrieval().load_data()


def test_duplicate_corpus_id_is_not_silently_overwritten(loader, release):
    rows = release["corpus_md", "corpus_md"].to_list()
    release["corpus_md", "corpus_md"] = Dataset.from_list(rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate corpus IDs"):
        NQTablesRetrieval().load_data()


def test_unknown_split_fails_before_download(loader):
    with pytest.raises(ValueError, match="Unsupported NQ-Tables splits"):
        NQTablesRetrieval().filter_eval_splits(["validation"]).load_data()
    loader.assert_not_called()
