import itertools

import numpy as np
import pytest

from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores, nauc

TOL = 1e-12


def test_nauc_order_invariant_for_equally_confident_queries():
    # regression for #5613: with constant confidence every abstention cutoff falls inside
    # the single tied confidence group, so the previous suffix-mean formula kept whichever
    # queries came first in the input. The same (confidence, metric) pairs scored +1.0 or
    # -1.0 depending on the insertion order, even though a constant confidence vector
    # carries no information and its abstention curve has to be flat (nAUC 0).
    confidence = np.ones(4)
    metrics = np.array([0.0, 0.0, 1.0, 1.0])

    assert nauc(confidence, metrics) == pytest.approx(0.0, abs=TOL)
    assert nauc(confidence, metrics[::-1]) == pytest.approx(0.0, abs=TOL)

    values = [
        float(nauc(confidence, np.array(permutation)))
        for permutation in itertools.permutations(metrics)
    ]
    assert max(values) - min(values) < TOL
    assert max(abs(value) for value in values) < TOL


def test_nauc_order_invariant_for_permuted_tied_queries():
    # permuting the queries permutes both vectors, so the nAUC may not move -- also when
    # only some of the confidences are tied and other cutoffs split no group at all.
    rng = np.random.default_rng(0)
    for _ in range(25):
        confidence = rng.integers(0, 3, size=12).astype(float)
        metrics = rng.random(12)
        values = [
            float(nauc(confidence[permutation], metrics[permutation]))
            for permutation in (rng.permutation(12) for _ in range(20))
        ]
        assert max(values) - min(values) < TOL, (confidence, metrics)


def test_nauc_is_the_expectation_over_tie_breakings():
    # resolving a confidence tie is not a symmetry the metric can distinguish, and the fix
    # defines the retained score as the expectation over those resolutions. Averaging
    # nauc() over every resolution of the tied group must therefore reproduce nauc().
    confidence = np.array([0.5, 0.5, 0.5, 0.2, 0.9])
    metrics = np.array([0.1, 0.4, 0.9, 0.2, 0.6])
    resolutions = [np.array([*tied, 3, 4]) for tied in itertools.permutations(range(3))]

    expected = np.mean([nauc(confidence[r], metrics[r]) for r in resolutions])
    assert nauc(confidence, metrics) == pytest.approx(expected, abs=TOL)


def test_nauc_unchanged_without_confidence_ties():
    # the fix must be a no-op for inputs without exact confidence ties; these are the
    # values the pre-#5613 implementation produced.
    confidence = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2])
    metrics = np.array([0.2, 0.8, 0.4, 0.9, 0.1, 0.6])
    rates = np.linspace(0.0, 0.8, 9)

    assert nauc(confidence, metrics) == pytest.approx(-0.836276083467094, abs=TOL)
    assert nauc(confidence, metrics, rates) == pytest.approx(
        -0.8568588469184883, abs=TOL
    )


def test_n_auc_invariant_to_query_order_in_retrieval_scores():
    # end-to-end regression for #5613 through the scoring path: four queries with two
    # candidates each, where every query has the same confidence under max, std and diff1.
    # The abstention curve is then flat, so every nAUC has to be 0, and it must not depend
    # on which of the two groups of queries retrieves the relevant document first.
    qrels = {f"q{i}": {"d_rel": 1} for i in range(4)}
    hits_first = {"d_rel": 1.0, "d_other": 0.0}
    hits_second = {"d_rel": 0.0, "d_other": 1.0}
    first_two_hit = {f"q{i}": (hits_first if i < 2 else hits_second) for i in range(4)}
    last_two_hit = {f"q{i}": (hits_second if i < 2 else hits_first) for i in range(4)}

    score_orders = [
        calculate_retrieval_scores(order, qrels, [1, 2, 3])
        for order in (first_two_hit, last_two_hit)
    ]
    per_query_ndcg = [
        [scores.all_scores[q]["ndcg_cut_1"] for q in qrels] for scores in score_orders
    ]
    assert per_query_ndcg == [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0]]

    for attribute in ("naucs", "naucs_mrr"):
        first, second = (getattr(scores, attribute) for scores in score_orders)
        assert first.keys() == second.keys()
        for key, value in first.items():
            if np.isnan(value):
                # a metric that is constant across the queries has a flat oracle curve,
                # so nAUC is undefined (nan) regardless of the query order
                assert np.isnan(second[key]), key
                continue
            # constant confidence gives a flat abstention curve, hence nAUC == 0
            assert value == pytest.approx(0.0, abs=TOL), key
            assert value == pytest.approx(second[key], abs=TOL), key
