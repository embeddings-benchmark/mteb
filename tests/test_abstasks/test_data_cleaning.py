"""Tests for `AbsTask.remove_duplicates` and `AbsTask.filter_short_documents`."""

import warnings
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict

from mteb.abstasks._data_filter.dataset_filters import (
    keep_first_occurrence,
    keep_long_enough,
    text_key,
)
from mteb.cache import ResultCache
from mteb.evaluate import OverwriteStrategy, _check_cache
from mteb.mocks import (
    MockClassificationTask,
    MockClusteringTask,
    MockImageClassificationTask,
    MockMultilabelClassification,
    MockMultilingualClassificationTask,
    MockPairClassificationTask,
    MockRegressionTask,
    MockRerankingTask,
    MockRetrievalTask,
)
from mteb.models.model_meta import ModelMeta
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
                    "text": [
                        "a shared text",
                        "a shared text ",
                        "hi",
                        "  ",
                        "long text",
                    ],
                    "label": [0, 1, 0, 1, 0],
                }
            ),
        }
    )
    task.data_loaded = True
    return task


def test_keep_first_occurrence_ignores_surrounding_whitespace() -> None:
    assert keep_first_occurrence("strip")([("a",), (" a ",), ("b",), ("a",)]) == [0, 2]


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

    task.remove_duplicates(normalize=normalize)  # type: ignore[arg-type]

    assert task.dataset["test"]["text"] == expected


def test_alphanumeric_drops_punctuation_instead_of_splitting_on_it() -> None:
    assert text_key(("e-mail",), "alphanumeric") == text_key(("email",), "alphanumeric")
    assert text_key(("e-mail",), "alphanumeric") == text_key(
        ("E-Mail",), "alphanumeric"
    )
    # whitespace is collapsed rather than removed, so word boundaries still tell texts apart
    assert text_key(("e mail",), "alphanumeric") != text_key(("email",), "alphanumeric")


@pytest.mark.parametrize(
    ("unit", "min_length", "expected"),
    [
        ("characters", 3, [1, 2]),
        ("words", 2, [2]),
    ],
)
def test_keep_long_enough(unit: str, min_length: int, expected: list[int]) -> None:
    texts = [("  ",), ("abc",), ("two words",)]

    assert keep_long_enough(min_length, unit)(texts) == expected  # type: ignore[arg-type]


def test_keep_long_enough_requires_every_column_to_be_long_enough() -> None:
    assert keep_long_enough(3, "characters")([("abc", "ab"), ("abc", "abc")]) == [1]


def test_remove_duplicates_keeps_first_occurrence_per_split() -> None:
    task = _classification_task()

    task.remove_duplicates()

    # "a shared text " is a duplicate of "a shared text" once stripped
    assert task.dataset["test"]["text"] == ["a shared text", "hi", "  ", "long text"]
    # duplicates are removed within a split, so the train occurrence is untouched
    assert task.dataset["train"]["text"] == ["a shared text", "train only"]


def test_remove_duplicates_can_be_restricted_to_a_split() -> None:
    task = _classification_task()

    task.remove_duplicates(splits=["train"])

    assert len(task.dataset["test"]) == 5


def test_filter_short_documents_defaults_to_characters() -> None:
    task = _classification_task()

    task.filter_short_documents(min_length=5)

    assert task.dataset["test"]["text"] == [
        "a shared text",
        "a shared text ",
        "long text",
    ]
    assert task.dataset["train"]["text"] == ["a shared text", "train only"]


def test_filter_short_documents_counts_words() -> None:
    task = _classification_task()

    task.filter_short_documents(min_length=3, unit="words")

    assert task.dataset["test"]["text"] == ["a shared text", "a shared text "]


def test_filters_keep_the_other_columns_aligned() -> None:
    task = _classification_task()

    task.remove_duplicates().filter_short_documents(min_length=5)

    assert task.dataset["test"]["label"] == [0, 0]


def test_filters_are_chainable_and_return_the_task() -> None:
    task = _classification_task()

    assert task.remove_duplicates() is task
    assert task.filter_short_documents() is task


def test_filters_load_the_data_when_it_is_not_loaded() -> None:
    task = MockClassificationTask()
    assert not task.data_loaded

    task.remove_duplicates()

    assert task.data_loaded


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


def test_filters_apply_to_every_subset_of_a_multilingual_task() -> None:
    task = _multilingual_task()

    task.remove_duplicates()

    for subset in task.dataset:
        assert task.dataset[subset]["test"]["text"] == ["duplicated"]


def test_filters_can_be_restricted_to_a_subset() -> None:
    task = _multilingual_task()

    task.remove_duplicates(subsets=["eng"])

    assert task.dataset["eng"]["test"]["text"] == ["duplicated"]
    assert task.dataset["fra"]["test"]["text"] == ["duplicated", "duplicated"]


def test_filters_use_both_columns_of_a_pair_task() -> None:
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

    task.remove_duplicates()

    # only the third row duplicates the first, the second differs in `sentence2`
    assert task.dataset["test"]["sentence2"] == ["other", "different"]


def test_filters_apply_within_a_row_of_a_clustering_task() -> None:
    task = MockClusteringTask()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentences": [["repeated", "repeated", "hi", "a long sentence"]],
                    "labels": [[0, 1, 2, 3]],
                }
            )
        }
    )
    task.data_loaded = True

    task.remove_duplicates().filter_short_documents(min_length=5)

    row = task.dataset["test"][0]
    assert row["sentences"] == ["repeated", "a long sentence"]
    # the labels of a cluster are filtered alongside its sentences
    assert row["labels"] == [0, 3]


def test_filters_raise_for_a_task_without_text() -> None:
    task = MockImageClassificationTask()

    with pytest.raises(NotImplementedError, match="does not know which columns"):
        task.remove_duplicates()


def test_filters_raise_when_the_selection_is_empty() -> None:
    task = _classification_task()

    with pytest.raises(ValueError, match="do not select any data"):
        task.remove_duplicates(splits=["not_a_split"])


def test_filters_accept_explicit_columns() -> None:
    task = _classification_task()

    with pytest.raises(ValueError, match="Cannot filter on"):
        task.remove_duplicates(columns=["not_a_column"])


def test_text_key_distinguishes_differently_split_rows() -> None:
    assert text_key(("a", "b")) != text_key(("ab", ""))
    assert text_key(("a", "b")) == text_key((" a ", "b "))


def test_a_filter_that_removes_nothing_leaves_the_task_unmodified() -> None:
    task = _classification_task()

    task.filter_short_documents(min_length=1, splits=["train"])

    assert not task.data_modified


def test_filtering_marks_the_task_as_modified_and_warns_once() -> None:
    task = _classification_task()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        task.remove_duplicates().filter_short_documents(min_length=5)

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
    task = _classification_task()

    with pytest.warns(UserWarning, match=r"splits \['test'\].*empty"):
        task.filter_short_documents(min_length=50, splits=["test"])


def test_a_label_that_filtering_made_untrainable_warns() -> None:
    task = MockClassificationTask()
    task.dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"text": ["keep me", "keep me too", "no"], "label": [0, 0, 1]}
            ),
            "test": Dataset.from_dict({"text": ["a test text"], "label": [1]}),
        }
    )
    task.data_loaded = True

    with pytest.warns(UserWarning, match="can never be predicted"):
        task.filter_short_documents(min_length=5)


def test_regression_and_multilabel_skip_the_label_checks() -> None:
    for task in (MockRegressionTask(), MockMultilabelClassification()):
        task.load_data()

        # the parent implementation would raise on a list label and be meaningless on a continuous one
        task._warn_about_label_distribution()


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

    task.remove_duplicates()

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

    task.remove_duplicates(normalize="alphanumeric")

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

    task.remove_duplicates()

    data = task.dataset[subset][split]
    assert data["queries"]["id"] == ["q1"]
    assert data["relevant_docs"] == {"q1": {"d1": 1, "d2": 1}}


def test_retrieval_short_filter_drops_judgements_and_orphaned_queries() -> None:
    task = MockRetrievalTask()
    task.load_data()
    subset, split = _retrieval_split(task)
    task.dataset[subset][split] = {
        "corpus": Dataset.from_dict(
            {"id": ["d1", "d2"], "text": ["a long document", "hi"]}
        ),
        "queries": Dataset.from_dict(
            {"id": ["q1", "q2"], "text": ["a long query", "another long query"]}
        ),
        "relevant_docs": {"q1": {"d1": 1}, "q2": {"d2": 1}},
        "top_ranked": None,
    }

    task.filter_short_documents(min_length=5)

    data = task.dataset[subset][split]
    assert data["corpus"]["id"] == ["d1"]
    # q2 only had the removed document as a positive, so it can no longer be scored
    assert data["queries"]["id"] == ["q1"]
    assert data["relevant_docs"] == {"q1": {"d1": 1}}


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

    task.remove_duplicates()

    data = task.dataset[subset][split]
    # d2 is remapped onto d1, which then already occurs in the list
    assert data["top_ranked"] == {"q1": ["d1", "d3"]}
