from __future__ import annotations

import pytest

from mteb.abstasks._statistics_calculation import (
    calculate_text_relevance_overlap_statistics,
)


@pytest.mark.parametrize(
    ("query", "document"),
    [
        ("CAFÉ!", "Un cafe\N{COMBINING ACUTE ACCENT}, s'il vous plaît."),
        ("北京天气", "今天的北京天气很好。"),
    ],
)
def test_text_relevance_overlap_normalizes_multilingual_text(
    query: str, document: str
) -> None:
    statistics = calculate_text_relevance_overlap_statistics(
        relevant_docs={"q1": {"d1": 1}},
        queries={"q1": query},
        corpus={"d1": document},
    )

    assert statistics == {
        "num_pairs": 1,
        "min_query_character_ngram_overlap": 1.0,
        "average_query_character_ngram_overlap": 1.0,
        "max_query_character_ngram_overlap": 1.0,
    }


def test_text_relevance_overlap_uses_character_ngrams_3_to_5() -> None:
    statistics = calculate_text_relevance_overlap_statistics(
        relevant_docs={"q1": {"d1": 1, "d2": 0}},
        queries={"q1": "abcd"},
        corpus={"d1": "abcx", "d2": "abcd"},
    )

    assert statistics == {
        "num_pairs": 1,
        "min_query_character_ngram_overlap": 1 / 3,
        "average_query_character_ngram_overlap": 1 / 3,
        "max_query_character_ngram_overlap": 1 / 3,
    }


def test_text_relevance_overlap_ignores_queries_without_3_grams() -> None:
    statistics = calculate_text_relevance_overlap_statistics(
        relevant_docs={"q1": {"d1": 1}},
        queries={"q1": "!?"},
        corpus={"d1": "anything"},
    )

    assert statistics is None


@pytest.mark.parametrize(
    ("queries", "corpus"),
    [
        ({}, {"d1": "document"}),
        ({"q1": "query"}, {}),
    ],
)
def test_text_relevance_overlap_ignores_missing_qrels_ids(
    queries: dict[str, str], corpus: dict[str, str]
) -> None:
    statistics = calculate_text_relevance_overlap_statistics(
        relevant_docs={"q1": {"d1": 1}},
        queries=queries,
        corpus=corpus,
    )

    assert statistics is None
