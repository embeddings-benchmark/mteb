from __future__ import annotations

from typing import Any

from datasets import Dataset

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.load_results.mteb_results import ScoresDict

from ..evaluation.evaluators import RerankingEvaluator
from .AbsTask import AbsDescriptiveStatistics, AbsTask


class RerankingDescriptiveStatistics(AbsDescriptiveStatistics):
    """Descriptive statistics for Reranking

    num_positive: Number of positive examples
    num_negative: Number of negative examples
    avg_query_len: Average length of queries
    avg_positive_len: Average length of positive examples
    avg_negative_len: Average length of negative examples
    """

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
        model: Encoder | EncoderWithQueryCorpusEncode,
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
        self, split: str, lang: str | None = None
    ) -> RerankingDescriptiveStatistics:
        if lang:
            query = self.dataset[lang][split]["query"]
            positive = self.dataset[lang][split]["positive"]
            negative = self.dataset[lang][split]["negative"]
        else:
            query = self.dataset[split]["query"]
            positive = self.dataset[split]["positive"]
            negative = self.dataset[split]["negative"]

        total_len_query = sum([len(q) for q in query])
        total_len_positive = sum([len(p) for p in positive])
        total_len_negative = sum([len(n) for n in negative])
        return {
            "num_samples": len(query),
            "num_positive": len(positive),
            "num_negative": len(negative),
            "avg_query_len": total_len_query / len(query),
            "avg_positive_len": total_len_positive / len(positive),
            "avg_negative_len": total_len_negative / len(negative),
        }
