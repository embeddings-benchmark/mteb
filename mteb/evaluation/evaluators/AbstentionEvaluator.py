from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class AbstentionEvaluator(Evaluator):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def __call__(self, model):
        """This is called during training to evaluate the model.
        It returns scores.

        Parameters
        ----------
        model:
            the model to evaluate
        """
        raise NotImplementedError("The abstention evaluator must not be called directly.")

    @staticmethod
    def compute_abstention_scores_retrieval(
        qrels: dict[str, dict[str, int]],
        results: dict[str, dict[str, float]],
    ) -> Dict[str, float]:

        top_hits = {}
        for query_id, doc_scores in results.items():
            top_hits[query_id] = sorted(
                doc_scores.items(), key=lambda item: item[1], reverse=True
            )[0:10]

        # TODO: Implement the abstention computation
        abstention_scores = {"Absention_Placeholder": 0.0}
        return abstention_scores
