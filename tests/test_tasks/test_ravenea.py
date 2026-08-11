import math

import pytest

from mteb.tasks.retrieval.eng.ravenea import (
    _source_gain_to_grade,
    _source_grade_to_gain,
    _source_metrics,
)


@pytest.mark.parametrize("grade", range(-3, 4))
def test_ravenea_gain_round_trip(grade: int) -> None:
    assert _source_gain_to_grade(_source_grade_to_gain(grade)) == grade


def test_ravenea_source_metrics() -> None:
    qrels = {
        "q1": {
            "best": _source_grade_to_gain(3),
            "good": _source_grade_to_gain(1),
            "bad": _source_grade_to_gain(-3),
        }
    }
    results = {"q1": {"good": 3.0, "best": 2.0, "bad": 1.0}}

    metrics = _source_metrics(qrels, results)

    assert metrics["ravenea_mrr"] == 0.5
    assert metrics["ravenea_precision_at_1"] == 0.0
    assert metrics["ravenea_precision_at_3"] == pytest.approx(1 / 3)
    expected_ndcg_at_3 = (
        _source_grade_to_gain(1) + _source_grade_to_gain(3) / math.log2(3)
    ) / (_source_grade_to_gain(3) + _source_grade_to_gain(1) / math.log2(3))
    assert metrics["ravenea_ndcg_at_3"] == pytest.approx(expected_ndcg_at_3)
