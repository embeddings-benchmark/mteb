from __future__ import annotations

from typing import Any, NamedTuple

HFSubset = (
    str  # e.g. 'en-de', 'en', 'default' (default is used when there is no subset)
)
SplitName = str  # e.g. 'test', 'validation', 'train'
Score = Any  # e.g. float, int, or any other type that represents a score, should be json serializable
ScoresDict = dict[
    str, Any
]  # e.g {'main_score': 0.5, 'hf_subset': 'en-de', 'languages': ['eng-Latn', 'deu-Latn']}


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
