from __future__ import annotations

import pytest

from mteb.abstasks._statistics_calculation import (
    calculate_text_corpus_overlap_statistics,
)


@pytest.mark.parametrize(
    ("query", "document"),
    [
        ("CAFÉ!", "Un cafe\N{COMBINING ACUTE ACCENT}, s'il vous plaît."),
        ("北京天气", "今天的北京天气很好。"),
    ],
)
def test_text_corpus_overlap_normalizes_multilingual_text(
    query: str, document: str
) -> None:
    statistics = calculate_text_corpus_overlap_statistics(
        queries={"q1": query},
        corpus={"d1": document},
    )

    assert statistics == {
        "num_queries": 1,
        "min_query_character_ngram_overlap": 1.0,
        "average_query_character_ngram_overlap": 1.0,
        "max_query_character_ngram_overlap": 1.0,
    }


def test_text_corpus_overlap_uses_character_4_grams() -> None:
    statistics = calculate_text_corpus_overlap_statistics(
        queries={"q1": "abcd"},
        corpus={"d1": "abcx"},
    )

    assert statistics == {
        "num_queries": 1,
        "min_query_character_ngram_overlap": 0.0,
        "average_query_character_ngram_overlap": 0.0,
        "max_query_character_ngram_overlap": 0.0,
    }


def test_text_corpus_overlap_ignores_queries_without_4_grams() -> None:
    statistics = calculate_text_corpus_overlap_statistics(
        queries={"q1": "abc"},
        corpus={"d1": "anything"},
    )

    assert statistics is None


def test_text_corpus_overlap_uses_all_queries_and_documents() -> None:
    statistics = calculate_text_corpus_overlap_statistics(
        queries={"q1": "abcd", "q2": "wxyz"},
        corpus={"d1": "abcx", "d2": "wxyz"},
    )

    assert statistics == {
        "num_queries": 2,
        "min_query_character_ngram_overlap": 0.0,
        "average_query_character_ngram_overlap": 0.5,
        "max_query_character_ngram_overlap": 1.0,
    }


def test_text_corpus_overlap_reports_zero_when_collection_has_no_match() -> None:
    statistics = calculate_text_corpus_overlap_statistics(
        queries={"q1": "abcd"},
        corpus={"d1": "wxyz"},
    )

    assert statistics == {
        "num_queries": 1,
        "min_query_character_ngram_overlap": 0.0,
        "average_query_character_ngram_overlap": 0.0,
        "max_query_character_ngram_overlap": 0.0,
    }


@pytest.mark.parametrize(
    ("queries", "corpus"),
    [({}, {"d1": "document"}), ({"q1": "query"}, {})],
)
def test_text_corpus_overlap_requires_queries_and_corpus(
    queries: dict[str, str], corpus: dict[str, str]
) -> None:
    statistics = calculate_text_corpus_overlap_statistics(
        queries=queries,
        corpus=corpus,
    )

    assert statistics is None
