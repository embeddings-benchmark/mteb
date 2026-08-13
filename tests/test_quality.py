"""Tests for `mteb.quality`."""

import warnings
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from mteb.cache import ResultCache
from mteb.evaluate import OverwriteStrategy, _check_cache
from mteb.mocks import (
    MockClassificationTask,
    MockClusteringTask,
    MockImageClassificationTask,
    MockMultilingualClassificationTask,
    MockPairClassificationTask,
    MockRerankingTask,
    MockRetrievalTask,
)
from mteb.models.model_meta import ModelMeta
from mteb.quality import remove_duplicates
from mteb.quality._apply import warn_about_unusable_data
from mteb.quality._row_filters import keep_first_occurrence, row_key
from mteb.results import TaskResult


def _classification_task() -> MockClassificationTask:
    task = MockClassificationTask()
    task.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": ["a shared text", "train only"], "label": [0, 1]}
            ),
            "test": Dataset.from_dict(
                {
                    "text": ["a shared text", "a shared text ", "hi", "long text"],
                    "label": [0, 1, 0, 0],
                }
            ),
        }
    )
    task.data_loaded = True
    return task


def test_keep_first_occurrence_returns_the_first_of_each_distinct_row() -> None:
    assert keep_first_occurrence([("a",), ("a",), ("b",), ("a",)]) == [0, 2]


def test_row_key_distinguishes_differently_split_rows() -> None:
    assert row_key(("a", "b")) != row_key(("ab", ""))


def test_remove_duplicates_keeps_first_occurrence_per_split() -> None:
    task = _classification_task()

    remove_duplicates(task)

    # "a shared text " is a duplicate of "a shared text" once stripped
    assert task.dataset["test"]["text"] == ["a shared text", "hi", "long text"]
    # duplicates are removed within a split, so the train occurrence is untouched
    assert task.dataset["train"]["text"] == ["a shared text", "train only"]


def test_remove_duplicates_keeps_the_other_columns_aligned() -> None:
    task = _classification_task()

    remove_duplicates(task)

    assert task.dataset["test"]["label"] == [0, 0, 0]


def test_remove_duplicates_returns_the_task() -> None:
    task = _classification_task()

    assert remove_duplicates(task) is task


def test_remove_duplicates_can_be_restricted_to_a_split() -> None:
    task = _classification_task()

    remove_duplicates(task, splits=["train"])

    assert len(task.dataset["test"]) == 4


def test_remove_duplicates_loads_the_data_when_it_is_not_loaded() -> None:
    task = MockClassificationTask()
    assert not task.data_loaded

    remove_duplicates(task)

    assert task.data_loaded


@pytest.mark.parametrize(
    ("normalize", "expected"),
    [
        ("strip", ["Wake me up!", "wake me up", "wake me up!", "wake  me  up"]),
        ("casefold", ["Wake me up!", "wake me up", "wake  me  up"]),
        ("alphanumeric", ["Wake me up!"]),
    ],
)
def test_normalize_controls_how_close_a_duplicate_has_to_be(
    normalize: str, expected: list[str]
) -> None:
    task = MockClassificationTask()
    texts = ["Wake me up!", "wake me up", "wake me up!", "wake  me  up"]
    task.dataset = DatasetDict(
        {"test": Dataset.from_dict({"text": texts, "label": [0] * len(texts)})}
    )
    task.data_loaded = True

    remove_duplicates(task, normalize=normalize)  # type: ignore[arg-type]

    assert task.dataset["test"]["text"] == expected


def test_alphanumeric_drops_punctuation_instead_of_splitting_on_it() -> None:
    task = MockClassificationTask()
    texts = ["e-mail", "email", "e mail"]
    task.dataset = DatasetDict(
        {"test": Dataset.from_dict({"text": texts, "label": [0, 1, 2]})}
    )
    task.data_loaded = True

    remove_duplicates(task, normalize="alphanumeric")

    # "e-mail" and "email" match, but whitespace is collapsed rather than removed
    assert task.dataset["test"]["text"] == ["e-mail", "e mail"]


def _multilingual_task() -> MockMultilingualClassificationTask:
    task = MockMultilingualClassificationTask()
    # the mock shares one DatasetDict between its subsets, so give each subset its own
    task.dataset = {
        subset: DatasetDict(
            {
                "test": Dataset.from_dict(
                    {"text": ["duplicated", "duplicated"], "label": [0, 1]}
                )
            }
        )
        for subset in ("eng", "fra")
    }
    task.data_loaded = True
    return task


def test_remove_duplicates_applies_to_every_subset() -> None:
    task = _multilingual_task()

    remove_duplicates(task)

    for subset in task.dataset:
        assert task.dataset[subset]["test"]["text"] == ["duplicated"]


def test_remove_duplicates_can_be_restricted_to_a_subset() -> None:
    task = _multilingual_task()

    remove_duplicates(task, subsets=["eng"])

    assert task.dataset["eng"]["test"]["text"] == ["duplicated"]
    assert task.dataset["fra"]["test"]["text"] == ["duplicated", "duplicated"]


def test_remove_duplicates_uses_both_columns_of_a_pair_task() -> None:
    task = MockPairClassificationTask()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentence1": ["same", "same", "same"],
                    "sentence2": ["other", "different", "other"],
                    "labels": [1, 0, 1],
                }
            )
        }
    )
    task.data_loaded = True

    remove_duplicates(task)

    # only the third row duplicates the first, the second differs in `sentence2`
    assert task.dataset["test"]["sentence2"] == ["other", "different"]


def test_remove_duplicates_applies_within_a_row_of_a_clustering_task() -> None:
    task = MockClusteringTask()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentences": [["repeated", "repeated", "distinct"]],
                    "labels": [[0, 1, 2]],
                }
            )
        }
    )
    task.data_loaded = True

    remove_duplicates(task)

    row = task.dataset["test"][0]
    assert row["sentences"] == ["repeated", "distinct"]
    # the labels of a cluster are filtered alongside its sentences
    assert row["labels"] == [0, 2]


def test_remove_duplicates_compares_images_by_content() -> None:
    task = MockImageClassificationTask()
    task.load_data()
    split = next(iter(task.metadata.eval_splits))
    images = task.dataset[split]["image"]
    task.dataset[split] = Dataset.from_dict(
        {"image": [images[0], images[0], images[1]], "label": [0, 1, 2]}
    )

    remove_duplicates(task)

    assert task.dataset[split]["label"] == [0, 2]


def test_remove_duplicates_raises_for_unknown_columns() -> None:
    task = _classification_task()

    with pytest.raises(ValueError, match="does not declare the columns"):
        remove_duplicates(task, columns=["not_a_column"])


def test_remove_duplicates_raises_when_nothing_is_selected() -> None:
    task = _classification_task()

    with pytest.raises(ValueError, match="do not select any data"):
        remove_duplicates(task, splits=["nope"])


def test_a_filter_that_removes_nothing_leaves_the_task_unmodified() -> None:
    task = _classification_task()

    remove_duplicates(task, splits=["train"])

    assert not task.data_modified


def test_filtering_marks_the_task_as_modified_and_warns_once() -> None:
    task = _classification_task()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        remove_duplicates(task)
        remove_duplicates(task, normalize="alphanumeric")

    assert task.data_modified
    # the second filter finds the task already modified and stays quiet about it
    assert sum("no longer matches revision" in str(w.message) for w in caught) == 1


def test_unloading_the_data_clears_the_modified_flag() -> None:
    task = _classification_task()
    task.data_modified = True

    task.unload_data()

    assert not task.data_modified


def test_a_modified_task_never_reads_the_results_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = _classification_task()
    loaded: list[str] = []

    cache = ResultCache(tmp_path)
    monkeypatch.setattr(
        ResultCache,
        "load_task_result",
        lambda _self, name, _meta: loaded.append(name),  # type: ignore[misc]
    )

    _check_cache(task, ModelMeta.create_empty(), cache, OverwriteStrategy.ONLY_MISSING)
    assert loaded == [task.metadata.name]

    task.data_modified = True
    _check_cache(task, ModelMeta.create_empty(), cache, OverwriteStrategy.ONLY_MISSING)
    assert loaded == [task.metadata.name]


def test_building_a_result_from_a_modified_task_warns() -> None:
    task = _classification_task()
    task.data_modified = True
    scores = {"test": {"default": {"accuracy": 1.0, "main_score": 1.0}}}

    with pytest.warns(UserWarning, match="modified locally"):
        TaskResult.from_task_results(task, scores, evaluation_time=1.0)


def test_an_emptied_split_warns() -> None:
    # deduplication alone cannot empty a split, but the staged filters can
    task = _classification_task()
    task.dataset["test"] = task.dataset["test"].select([])

    with pytest.warns(UserWarning, match=r"splits \['test'\].*empty"):
        warn_about_unusable_data(task)


def test_a_label_that_filtering_made_untrainable_warns() -> None:
    task = MockClassificationTask()
    task.dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["dup", "dup"], "label": [0, 1]}),
            "test": Dataset.from_dict({"text": ["a test text"], "label": [1]}),
        }
    )
    task.data_loaded = True

    # deduplicating train drops the only example of label 1
    with pytest.warns(UserWarning, match="can never be predicted"):
        remove_duplicates(task, splits=["train"])


def _retrieval_split(task: MockRetrievalTask) -> tuple[str, str]:
    subset = next(iter(task.dataset))
    return subset, next(iter(task.dataset[subset]))


def test_retrieval_deduplication_moves_judgements_to_the_kept_document() -> None:
    task = MockRetrievalTask()
    task.load_data()
    subset, split = _retrieval_split(task)
    task.dataset[subset][split] = {
        "corpus": Dataset.from_dict(
            {"id": ["d1", "d2", "d3"], "text": ["same doc", "same doc", "other doc"]}
        ),
        "queries": Dataset.from_dict({"id": ["q1", "q2"], "text": ["q one", "q two"]}),
        "relevant_docs": {"q1": {"d1": 1}, "q2": {"d2": 1, "d3": 1}},
        "top_ranked": None,
    }

    remove_duplicates(task)

    data = task.dataset[subset][split]
    assert data["corpus"]["id"] == ["d1", "d3"]
    # q2's judgement for the removed duplicate d2 now points at d1
    assert data["relevant_docs"] == {"q1": {"d1": 1}, "q2": {"d1": 1, "d3": 1}}


def test_retrieval_remap_uses_the_same_normalization_as_the_filter() -> None:
    task = MockRetrievalTask()
    task.load_data()
    subset, split = _retrieval_split(task)
    task.dataset[subset][split] = {
        "corpus": Dataset.from_dict(
            {"id": ["d1", "d2"], "text": ["Same doc!", "same  doc"]}
        ),
        "queries": Dataset.from_dict({"id": ["q1"], "text": ["a query"]}),
        "relevant_docs": {"q1": {"d2": 1}},
        "top_ranked": None,
    }

    remove_duplicates(task, normalize="alphanumeric")

    data = task.dataset[subset][split]
    assert data["corpus"]["id"] == ["d1"]
    # d2 is only a duplicate under this normalization, so the remap has to use it too
    assert data["relevant_docs"] == {"q1": {"d1": 1}}


def test_retrieval_deduplication_merges_duplicated_queries() -> None:
    task = MockRetrievalTask()
    task.load_data()
    subset, split = _retrieval_split(task)
    task.dataset[subset][split] = {
        "corpus": Dataset.from_dict(
            {"id": ["d1", "d2"], "text": ["first doc", "second doc"]}
        ),
        "queries": Dataset.from_dict(
            {"id": ["q1", "q2"], "text": ["same query", "same query"]}
        ),
        "relevant_docs": {"q1": {"d1": 1}, "q2": {"d2": 1}},
        "top_ranked": None,
    }

    remove_duplicates(task)

    data = task.dataset[subset][split]
    assert data["queries"]["id"] == ["q1"]
    assert data["relevant_docs"] == {"q1": {"d1": 1, "d2": 1}}


def test_reranking_top_ranked_is_kept_consistent() -> None:
    task = MockRerankingTask()
    task.load_data()
    subset, split = _retrieval_split(task)
    task.dataset[subset][split] = {
        "corpus": Dataset.from_dict(
            {"id": ["d1", "d2", "d3"], "text": ["same doc", "same doc", "other doc"]}
        ),
        "queries": Dataset.from_dict({"id": ["q1"], "text": ["a query"]}),
        "relevant_docs": {"q1": {"d1": 1}},
        "top_ranked": {"q1": ["d2", "d3", "d1"]},
    }

    remove_duplicates(task)

    data = task.dataset[subset][split]
    # d2 is remapped onto d1, which then already occurs in the list
    assert data["top_ranked"] == {"q1": ["d1", "d3"]}
