import math

import pytest

from mteb._evaluators.retrieval_metrics import ndcg_float_scores

# mteb rounds metric means to 5 decimals
TOL = 1e-5


def test_distinct_scores_match_manual_ndcg():
    # identity gains, discount 1/log2(rank + 1); the ideal uses the query's best
    # gains regardless of what the model retrieved.
    gains = {"q1": {"d1": 1.0, "d2": 0.5, "d3": 0.0}}
    results = {"q1": {"d2": 0.9, "d1": 0.7, "d3": 0.1}}  # d2 ranked above d1

    scores = ndcg_float_scores(gains, results, [2])

    dcg = 0.5 + 1.0 / math.log2(3)
    idcg = 1.0 + 0.5 / math.log2(3)
    assert scores["ndcg_float_at_2"] == pytest.approx(dcg / idcg, abs=TOL)


def test_exact_ties_credit_group_mean_gain():
    # all three scores tie, so the ranking asserts nothing: each position is
    # credited the group-mean gain (1 + 0 + 0) / 3 -- the expectation over all
    # tie resolutions.
    gains = {"q1": {"d1": 1.0, "d2": 0.0, "d3": 0.0}}
    results = {"q1": {"d1": 0.5, "d2": 0.5, "d3": 0.5}}

    scores = ndcg_float_scores(gains, results, [10])

    expected = (1 / 3) * (1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4))
    assert scores["ndcg_float_at_10"] == pytest.approx(expected, abs=TOL)


def test_tie_block_crossing_k_boundary():
    # d2 and d3 tie across the cutoff: each is credited half of the block's
    # gains, so the score does not depend on which one lands inside the cutoff.
    gains = {"q1": {"d1": 0.0, "d2": 1.0, "d3": 0.0}}
    results = {"q1": {"d1": 1.0, "d2": 0.5, "d3": 0.5}}

    scores = ndcg_float_scores(gains, results, [2])

    expected = (0.5 / math.log2(3)) / 1.0  # d1 contributes 0 at rank 1
    assert scores["ndcg_float_at_2"] == pytest.approx(expected, abs=TOL)


def test_unjudged_document_scores_zero_gain():
    gains = {"q1": {"d1": 1.0}}
    results = {"q1": {"d1": 0.9, "d_unjudged": 0.8}}

    scores = ndcg_float_scores(gains, results, [2])

    # the unjudged doc ranks second with gain 0; the ideal still uses d1's 1.0
    assert scores["ndcg_float_at_2"] == pytest.approx(1.0, abs=TOL)


def test_all_zero_gains_score_zero_without_dividing_by_zero():
    gains = {"q1": {"d1": 0.0, "d2": 0.0}}
    results = {"q1": {"d1": 0.9, "d2": 0.1}}

    scores = ndcg_float_scores(gains, results, [10])

    assert scores["ndcg_float_at_10"] == 0.0


@pytest.mark.parametrize("bad_gain", [-0.1, float("nan"), float("inf")])
def test_non_finite_or_negative_gains_raise(bad_gain: float):
    # `nan < 0` is False, so a finiteness check has to precede the sign check;
    # without it a NaN or inf gain reaches the mean and publishes `nan`.
    gains = {"q1": {"d1": bad_gain, "d2": 0.5}}
    results = {"q1": {"d1": 0.9, "d2": 0.1}}

    with pytest.raises(ValueError, match="finite and non-negative"):
        ndcg_float_scores(gains, results, [10])


def test_nan_model_score_raises_instead_of_scoring():
    # a NaN model score is a model bug: it must fail loudly. Without the guard,
    # a NaN becomes its own singleton tie class and the query silently scores
    # 1.0 (the committed 2545f4e code crashed with ZeroDivisionError on the
    # same input -- neither behaviour is designed; the guard pins the designed
    # one).
    gains = {"q1": {"d1": 1.0, "d2": 0.5}}
    results = {"q1": {"d1": float("nan"), "d2": 0.5}}

    with pytest.raises(ValueError, match="NaN model score"):
        ndcg_float_scores(gains, results, [10])

    # inf scores are well-ordered and scored like any large finite score
    scores = ndcg_float_scores(gains, {"q1": {"d1": float("inf"), "d2": 0.5}}, [10])
    assert scores["ndcg_float_at_10"] == pytest.approx(1.0, abs=TOL)


def test_multiple_k_values_and_naucs_keys():
    gains = {
        "q1": {"d1": 1.0, "d2": 0.5},
        "q2": {"d1": 0.2, "d2": 0.8},
    }
    results = {
        "q1": {"d1": 0.9, "d2": 0.1},
        "q2": {"d1": 0.3, "d2": 0.8},
    }

    scores = ndcg_float_scores(gains, results, [1, 10])

    # both models rank their highest-gain doc first on both queries
    assert scores["ndcg_float_at_1"] == pytest.approx(1.0, abs=TOL)
    assert scores["ndcg_float_at_10"] == pytest.approx(1.0, abs=TOL)
    assert "nauc_ndcg_float_at_10_max" in scores
    assert "nauc_ndcg_float_at_10_diff1" in scores
