from __future__ import annotations

from typing import Any

from datasets import Dataset

from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import ScoresDict

from ..evaluation.evaluators import RerankingEvaluator
from .AbsTask import AbsTask, DescriptiveStatistics


class RerankingDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Reranking

    Attributes:
        num_samples: number of samples in the dataset.
        num_positive: Number of positive examples
        num_negative: Number of negative examples
        avg_query_len: Average length of queries
        avg_positive_len: Average length of positive examples
        avg_negative_len: Average length of negative examples
    """

    num_samples: int
    num_positive: int
    num_negative: int
    avg_query_len: float
    avg_positive_len: float
    avg_negative_len: float


class AbsTaskReranking(AbsTask):
    """Abstract class for re-ranking experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        query: str
        positive: list[str]
        negative: list[str]
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> ScoresDict:
        evaluator = RerankingEvaluator(
            data_split,
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        scores = evaluator(model)

        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RerankingDescriptiveStatistics:
        if hf_subset:
            query = self.dataset[hf_subset][split]["query"]
            positive = self.dataset[hf_subset][split]["positive"]
            negative = self.dataset[hf_subset][split]["negative"]
        elif compute_overall:
            query = []
            positive = []
            negative = []
            for hf_subset in self.metadata.eval_langs:
                query.extend(self.dataset[hf_subset][split]["query"])
                positive.extend(self.dataset[hf_subset][split]["positive"])
                negative.extend(self.dataset[hf_subset][split]["negative"])
        else:
            query = self.dataset[split]["query"]
            positive = self.dataset[split]["positive"]
            negative = self.dataset[split]["negative"]

        total_len_query = sum([len(q) for q in query])
        total_len_positive = sum([len(p) for p in positive])
        total_len_negative = sum([len(n) for n in negative])
        return RerankingDescriptiveStatistics(
            num_samples=len(query),
            num_positive=len(positive),
            num_negative=len(negative),
            avg_query_len=total_len_query / len(query),
            avg_positive_len=total_len_positive / len(positive),
            avg_negative_len=total_len_negative / len(negative),
        )
