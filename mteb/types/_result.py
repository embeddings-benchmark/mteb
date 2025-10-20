from typing import Any, NamedTuple

HFSubset = str
"""The name of a HuggingFace dataset subset, e.g. 'en-de', 'en', 'default' (default is used when there is no subset)."""
SplitName = str
"""The name of a data split, e.g. 'test', 'validation', 'train'."""

Score = Any
"""A score value, could e.g. be accuracy. Normally it is a float or int, but it can take on any value. Should be json serializable."""

ScoresDict = dict[str, Score]
"""A dictionary of scores, typically also include metadata, e.g {'main_score': 0.5, 'accuracy': 0.5, 'f1': 0.6, 'hf_subset': 'en-de', 'languages': ['eng-Latn', 'deu-Latn']}"""


class RetrievalEvaluationResult(NamedTuple):
    """Holds the results of retrieval evaluation metrics."""

    all_scores: dict[str, dict[str, float]]
    ndcg: dict[str, float]
    map: dict[str, float]
    recall: dict[str, float]
    precision: dict[str, float]
    naucs: dict[str, float]
    mrr: dict[str, float]
    naucs_mrr: dict[str, float]
    cv_recall: dict[str, float]
