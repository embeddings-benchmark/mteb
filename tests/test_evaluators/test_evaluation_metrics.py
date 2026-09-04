import pytrec_eval

from mteb._evaluators.retrieval_metrics import (
    calculate_pmrr,
    calculate_retrieval_scores,
    mrr,
    recall_cap,
)


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


def test_skip_first_result_tiebreak_independent_of_insertion_order():
    # regression for the same #5092 tie-break bug in skip_first_result: the top hit is
    # dropped with a score-only stable sort, so when the top two scores tie it was dict
    # insertion order that decided which doc got discarded. Both runs below carry
    # identical scores, so they must score identically.
    qrels = {"q1": {"d_dup": 1, "d_other": 1}}
    tied_self_first = {"q1": {"d_self": 1.0, "d_dup": 1.0, "d_other": 0.5}}
    tied_dup_first = {"q1": {"d_dup": 1.0, "d_self": 1.0, "d_other": 0.5}}

    self_first = calculate_retrieval_scores(tied_self_first, qrels, [1, 2], True)
    dup_first = calculate_retrieval_scores(tied_dup_first, qrels, [1, 2], True)

    assert self_first.ndcg == dup_first.ndcg
    assert self_first.recall == dup_first.recall


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
