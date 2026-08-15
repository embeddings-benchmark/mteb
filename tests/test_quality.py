"""Tests for `mteb.quality`."""

import logging

import pytest
from datasets import Dataset, DatasetDict

import mteb
from mteb.evaluate import _check_data_not_modified
from mteb.mocks import (
    MockClassificationTask,
    MockClusteringTask,
    MockImageClassificationTask,
    MockMultilingualClassificationTask,
    MockPairClassificationTask,
    MockRerankingTask,
    MockRetrievalTask,
    MockSTSTask,
)
from mteb.quality import remove_duplicates
from mteb.quality._filters import (
    _keep_first_occurrence,
    _row_key,
    _warn_about_unusable_data,
)
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
    assert _keep_first_occurrence([("a",), ("a",), ("b",), ("a",)]) == [0, 2]


def test_row_key_distinguishes_differently_split_rows() -> None:
    assert _row_key(("a", "b")) != _row_key(("ab", ""))


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


def test_filtering_marks_the_task_as_modified_and_reports_it_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = _classification_task()

    with caplog.at_level(logging.WARNING, logger="mteb.quality._filters"):
        remove_duplicates(task)
        remove_duplicates(task, normalize="alphanumeric")

    assert task.data_modified
    # the second filter finds the task already modified and stays quiet about it
    assert sum("no longer matches revision" in m for m in caplog.messages) == 1


def test_unloading_the_data_clears_the_modified_flag() -> None:
    task = _classification_task()
    task.data_modified = True

    task.unload_data()

    assert not task.data_modified


def test_a_modified_task_cannot_be_evaluated() -> None:
    task = remove_duplicates(_classification_task())
    model = mteb.get_model("mteb/baseline-random-encoder")

    with pytest.raises(ValueError, match="modified locally"):
        mteb.evaluate(model, [task], cache=None, co2_tracker=False)


def test_an_unmodified_task_passes_the_guard() -> None:
    _check_data_not_modified([_classification_task()])


def test_building_a_result_from_a_modified_task_warns() -> None:
    task = _classification_task()
    task.data_modified = True
    scores = {"test": {"default": {"accuracy": 1.0, "main_score": 1.0}}}

    with pytest.warns(UserWarning, match="modified locally"):
        TaskResult.from_task_results(task, scores, evaluation_time=1.0)


def test_an_emptied_split_is_reported(caplog: pytest.LogCaptureFixture) -> None:
    # deduplication alone cannot empty a split, but the staged filters can
    task = _classification_task()
    task.dataset["test"] = task.dataset["test"].select([])

    with caplog.at_level(logging.WARNING, logger="mteb.quality._filters"):
        _warn_about_unusable_data(task)

    assert any("splits ['test']" in m and "empty" in m for m in caplog.messages)


def test_a_label_that_filtering_made_untrainable_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    task = MockClassificationTask()
    task.dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["dup", "dup"], "label": [0, 1]}),
            "test": Dataset.from_dict({"text": ["a test text"], "label": [1]}),
        }
    )
    task.data_loaded = True

    # deduplicating train drops the only example of label 1
    with caplog.at_level(logging.WARNING, logger="mteb.quality._classification"):
        remove_duplicates(task, splits=["train"])

    assert any("can never be predicted" in m for m in caplog.messages)


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


def test_an_unsupported_modality_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    task = _classification_task()
    monkeypatch.setattr(
        type(task), "_get_content_columns", lambda _self: {"text": "smell"}
    )

    with pytest.raises(NotImplementedError, match="cannot compare the \\['smell'\\]"):
        remove_duplicates(task)


def test_swapped_pairs_are_duplicates_for_a_symmetric_task() -> None:
    task = MockSTSTask()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentence1": ["alpha", "beta", "gamma"],
                    "sentence2": ["beta", "alpha", "delta"],
                    "score": [1.0, 3.0, 2.0],
                }
            )
        }
    )
    task.data_loaded = True

    remove_duplicates(task)

    # ("beta", "alpha") is the same pair as ("alpha", "beta"), so only the first is kept
    assert task.dataset["test"]["sentence1"] == ["alpha", "gamma"]
    assert task.dataset["test"]["score"] == [1.0, 2.0]


def test_swapped_pairs_are_distinct_for_an_order_sensitive_task() -> None:
    task = MockPairClassificationTask()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentence1": ["alpha", "beta"],
                    "sentence2": ["beta", "alpha"],
                    "labels": [1, 0],
                }
            )
        }
    )
    task.data_loaded = True

    remove_duplicates(task)

    assert len(task.dataset["test"]) == 2


def test_narrowing_to_one_side_keeps_the_comparison_order_sensitive() -> None:
    task = MockSTSTask()
    task.dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "sentence1": ["alpha", "beta"],
                    "sentence2": ["beta", "alpha"],
                    "score": [1.0, 3.0],
                }
            )
        }
    )
    task.data_loaded = True

    remove_duplicates(task, columns=["sentence1"])

    assert len(task.dataset["test"]) == 2
