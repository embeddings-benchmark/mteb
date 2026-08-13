from __future__ import annotations

import pytest

from mteb.tasks.retrieval.zxx.evve_retrieval import (
    evve_scores,
    trapezoidal_average_precision,
)


def test_trapezoidal_average_precision_matches_evve_evaluator() -> None:
    ranking = ["negative", "positive-a", "ignored", "positive-b"]

    score = trapezoidal_average_precision(
        ranking,
        {"positive-a", "positive-b"},
        {"ignored"},
    )

    # Positive ranks after removing the ignored item are [1, 2].
    # EVVE integrates trapezoids: (0 + 1/2)/4 + (1/2 + 2/3)/4.
    assert score == pytest.approx(5 / 12)


def test_evve_scores_balance_events_instead_of_queries() -> None:
    qrels = {
        "q1": {"a": 1},
        "q2": {"a": 1},
        "q3": {"b": 1},
    }
    results = {
        "q1": {"a": 2.0, "b": 1.0},
        "q2": {"a": 2.0, "b": 1.0},
        "q3": {"a": 2.0, "b": 1.0},
    }

    scores = evve_scores(
        qrels,
        results,
        {"q1": "event-a", "q2": "event-a", "q3": "event-b"},
        {"q1": (), "q2": (), "q3": ()},
    )

    # Trapezoidal AP is 1.0 for q1/q2 and 0.25 for q3.
    assert scores["evve_overall_map"] == pytest.approx(0.75)
    assert scores["evve_avg_map"] == pytest.approx(0.625)
