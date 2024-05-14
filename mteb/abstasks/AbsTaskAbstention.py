from __future__ import annotations

import logging

from ..evaluation.evaluators import (AbstentionRerankingEvaluator,
                                     AbstentionRetrievingEvaluator)
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskAbstention(AbsTask):
    """Abstract class for Abstention experiments.

    Abstention tasks are multi-inheritance tasks that inherit from a base task as well as the abstention task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate_monolingual(
        self, retriever, corpus, queries, relevant_docs, lang=None, **kwargs
    ):
        """Function to override for retrieval tasks"""
        if self.abstention_task == "Retrieval":
            return AbstentionRetrievingEvaluator(
                metadata_dict=self.metadata_dict
            ).evaluate_monolingual_retrieval_abstention(
                retriever, corpus, queries, relevant_docs, lang, **kwargs
            )
        raise NotImplementedError("Abstention task not defined.")

    def evaluate(self, model, split="test", **kwargs):
        if self.abstention_task == "Reranking":
            if not self.data_loaded:
                self.load_data()

            scores = {}
            if self.is_multilingual:
                for lang in self.langs:
                    data_split = self.dataset[lang][split]
                    evaluator = AbstentionRerankingEvaluator(data_split, **kwargs)
                    scores[lang] = evaluator(model)
            else:
                data_split = self.dataset[split]

                evaluator = AbstentionRerankingEvaluator(data_split, **kwargs)
                scores = evaluator(model)

            return dict(scores)

        if self.abstention_task == "Retrieval":
            return super().evaluate(model, split="test", **kwargs)

        raise NotImplementedError("Abstention task not defined.")
