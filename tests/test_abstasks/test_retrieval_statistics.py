"""Tests for qrel-ID membership statistics (see issue #5170).

`calculate_relevant_docs_statistics` reports how many unique qrel query/corpus
IDs are absent from the loaded query/corpus splits.
"""

from mteb.abstasks._statistics_calculation import calculate_relevant_docs_statistics


def test_no_missing_ids() -> None:
    relevant_docs = {"q1": {"d1": 1, "d2": 1}, "q2": {"d2": 1}}
    stats = calculate_relevant_docs_statistics(
        relevant_docs, query_ids={"q1", "q2"}, corpus_ids={"d1", "d2"}
    )
    assert stats["num_missing_query_ids"] == 0
    assert stats["num_missing_corpus_ids"] == 0


def test_missing_query_id() -> None:
    relevant_docs = {"q1": {"d1": 1}, "q_missing": {"d1": 1}}
    stats = calculate_relevant_docs_statistics(
        relevant_docs, query_ids={"q1"}, corpus_ids={"d1"}
    )
    assert stats["num_missing_query_ids"] == 1
    assert stats["num_missing_corpus_ids"] == 0


def test_missing_corpus_id() -> None:
    relevant_docs = {"q1": {"d1": 1, "d_missing": 1}}
    stats = calculate_relevant_docs_statistics(
        relevant_docs, query_ids={"q1"}, corpus_ids={"d1"}
    )
    assert stats["num_missing_query_ids"] == 0
    assert stats["num_missing_corpus_ids"] == 1


def test_missing_corpus_id_with_score_zero_is_counted() -> None:
    # Dangling references are counted regardless of relevance score:
    # a qrel-referenced corpus ID absent from the corpus is reported even at score 0.
    relevant_docs = {"q1": {"d1": 1, "d_missing": 0}}
    stats = calculate_relevant_docs_statistics(
        relevant_docs, query_ids={"q1"}, corpus_ids={"d1"}
    )
    assert stats["num_missing_corpus_ids"] == 1


def test_repeated_missing_ids_are_counted_once() -> None:
    # d_missing is referenced by two queries; both query IDs are missing.
    # Counts must be UNIQUE IDs, not qrel rows.
    relevant_docs = {
        "q_missing": {"d_missing": 1, "d1": 1},
        "q_missing_2": {"d_missing": 1},
    }
    stats = calculate_relevant_docs_statistics(
        relevant_docs, query_ids={"q1"}, corpus_ids={"d1"}
    )
    assert stats["num_missing_query_ids"] == 2  # q_missing, q_missing_2
    assert stats["num_missing_corpus_ids"] == 1  # d_missing (deduplicated)
