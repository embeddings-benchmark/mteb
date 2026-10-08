import math

import pytest
import pytrec_eval

from mteb._evaluators.retrieval_metrics import (
    calculate_pmrr,
    mrr,
    ndcg_float_scores,
    recall_cap,
)

# tolerance for ndcg_float_scores: mteb rounds metric means to 5 decimals
TOL = 1e-5


def test_recall_cap_no_relevant_docs_yields_none():
    # a query whose qrels hold only non-relevant (relevance 0) judgments has an empty
    # relevant set, so the R_cap denominator is 0. The zero guard should record None and
    # skip the division; without the skip it fell through and raised ZeroDivisionError.
    qrels = {"q1": {"d1": 0, "d2": 0}}
    results = {"q1": {"d1": 0.9, "d2": 0.1}}

    assert recall_cap(qrels, results, [10]) == {"R_cap_at_10": [None]}


def test_recall_cap_counts_capped_relevant_hits():
    # normal path stays intact: 2 relevant docs retrieved, capped at min(#relevant, k).
    qrels = {"q1": {"d1": 1, "d2": 1, "d3": 0}}
    results = {"q1": {"d1": 0.9, "d2": 0.8, "d3": 0.1}}

    assert recall_cap(qrels, results, [10]) == {"R_cap_at_10": [1.0]}


def test_mrr_tiebreak_independent_of_insertion_order():
    # regression for #5092: with tied scores a stable score-only sort ranked docs by
    # dict insertion order, so a constant scorer scored MRR@10 = 1.0 on tasks that list
    # the positive candidate first.
    qrels = {"q1": {"d_pos": 1}}
    tied = {"d_pos": 0.5, "d_x": 0.5, "d_y": 0.5}

    forward = mrr(qrels, {"q1": tied}, [10])["MRR@10"][0]
    reversed_order = mrr(qrels, {"q1": dict(reversed(tied.items()))}, [10])["MRR@10"][0]
    assert forward == reversed_order


def test_mrr_tiebreak_matches_pytrec_eval():
    # MRR must break ties the same way as the pytrec_eval metrics (by doc id, descending)
    # so a single ranking backs every metric. Positive is listed first but has the
    # lowest doc id, so insertion order and the correct order disagree.
    qrels = {"q1": {"d_a": 1}}
    results = {"q1": {"d_a": 0.5, "d_x": 0.5, "d_y": 0.5}}

    got = mrr(qrels, results, [10])["MRR@10"][0]
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"})
    expected = evaluator.evaluate(results)["q1"]["recip_rank"]

    assert got == expected == 1 / 3


def test_p_mrr_tiebreak_independent_of_insertion_order():
    # regression for the same #5092 tie-break bug in get_rank_from_dict: p-MRR ranks a
    # doc with a score-only stable sort, so on tied scores its rank followed dict
    # insertion order. Here both runs carry identical scores, so p-MRR must be 0 — but
    # the reversed insertion order moves the changed doc from rank 1 to rank 3 without
    # the fix, producing a spurious non-zero change.
    changed_qrels = {"a": ["0"]}
    tied = {"0": 0.5, "1": 0.5, "2": 0.5}

    original_run = {"a-og": tied}
    new_run = {"a-changed": dict(reversed(tied.items()))}

    score = calculate_pmrr(original_run, new_run, changed_qrels)
    assert score == 0.0


def test_p_mrr():
    changed_qrels = {
        "a": ["0"],
    }

    # these are the query: {"doc_id": score}
    original_run = {
        "a-og": {"0": 1, "1": 2, "2": 3, "3": 4},
    }

    new_run = {
        "a-changed": {"0": 1, "1": 2, "2": 3, "3": 4},
    }

    score = calculate_pmrr(
        original_run,
        new_run,
        changed_qrels,
    )
    assert score == 0.0

    # test with a change
    new_run = {
        "a-changed": {"0": 4, "1": 1, "2": 2, "3": 3},
    }

    score = calculate_pmrr(
        original_run,
        new_run,
        changed_qrels,
    )
    assert score == -0.75

    # test with a positive change, flipping them
    new_run = {
        "a-og": {"0": 4, "1": 1, "2": 2, "3": 3},
    }
    original_run = {
        "a-changed": {"0": 1, "1": 2, "2": 3, "3": 4},
    }
    score = calculate_pmrr(
        new_run,
        original_run,
        changed_qrels,
    )
    assert score == 0.75


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


def test_query_without_gains_scores_zero_and_stays_in_the_mean():
    # q2 has no gains at all (e.g. its qrels gain entries are all null): it scores
    # 0.0 and still counts in the mean instead of being dropped from it.
    gains = {"q1": {"d1": 1.0, "d2": 0.0}}
    results = {"q1": {"d1": 0.9, "d2": 0.1}, "q2": {"d1": 0.9, "d2": 0.1}}

    scores = ndcg_float_scores(gains, results, [10])

    assert scores["ndcg_float_at_10"] == pytest.approx(0.5, abs=TOL)


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
    # 1.0, or the sort crashes, depending on the input.
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
