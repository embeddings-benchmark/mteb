from mteb._evaluators.retrieval_metrics import calculate_pmrr


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
