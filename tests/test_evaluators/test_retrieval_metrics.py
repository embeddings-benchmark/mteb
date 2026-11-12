import numpy as np

from mteb._evaluators.retrieval_metrics import nauc


def test_nauc_is_the_same_for_every_order_of_equally_confident_queries():
    """A confidence tie is not a signal, so the order of equally confident queries must not change nAUC.

    With a constant confidence vector every abstention cutoff falls inside the single tied
    group. Before #5613 the retained scores were taken from a suffix of the sorted scores,
    so which queries a cutoff dropped depended on the input order: the same two vectors
    scored +1.0 when the queries retrieving the relevant document came first and
    -1.0000000000000002 when they came last, even though a constant confidence carries no
    information and the abstention curve has to be flat.
    """
    confidence = np.ones(4)
    metrics = np.array([0.0, 0.0, 1.0, 1.0])

    assert nauc(confidence, metrics) == 0.0
    assert nauc(confidence, metrics[::-1]) == 0.0
