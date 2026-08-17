import pytrec_eval

from mteb._evaluators.retrieval_metrics import calculate_pmrr, mrr


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
