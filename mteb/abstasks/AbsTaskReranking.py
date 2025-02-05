from __future__ import annotations

import warnings
from typing import Any

from datasets import Dataset

from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import ScoresDict

from ..evaluation.evaluators import RerankingEvaluator
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics


class RerankingDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Reranking

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        num_positive: Number of positive examples
        num_negative: Number of negative examples

        min_query_length: Minimum length of queries
        avg_query_length: Average length of queries
        max_query_length: Maximum length of queries
        unique_query: Number of unique queries

        min_positive_length: Minimum length of positive examples
        avg_positive_length: Average length of positive examples
        max_positive_length: Maximum length of positive examples
        unique_positive: Number of unique positive examples

        min_negative_length: Minimum length of negative examples
        avg_negative_length: Average length of negative examples
        max_negative_length: Maximum length of negative examples
        unique_negative: Number of unique negative examples
    """

    num_samples: int
    number_of_characters: int
    num_positive: int
    num_negative: int

    min_query_length: int
    avg_query_length: float
    max_query_length: int
    unique_query: int

    min_positive_length: int
    avg_positive_length: float
    max_positive_length: int
    unique_positive: int

    min_negative_length: int
    avg_negative_length: float
    max_negative_length: int
    unique_negative: int


class AbsTaskReranking(AbsTask):
    """Abstract class for re-ranking experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        query: str
        positive: list[str]
        negative: list[str]
    """

    abstask_prompt = "Retrieve text based on user query."

    def __init__(self, **kwargs):
        warnings.warn(
            "`AbsTaskReranking` will be merged with AbsTaskRetrieval in v2.0.0.",
            DeprecationWarning,
        )
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
            positive = transform_reranking_data(
                self.dataset[hf_subset][split]["positive"]
            )
            negative = transform_reranking_data(
                self.dataset[hf_subset][split]["negative"]
            )
        elif compute_overall:
            query = []
            positive = []
            negative = []
            for hf_subset in self.metadata.eval_langs:
                query.extend(self.dataset[hf_subset][split]["query"])
                positive.extend(
                    transform_reranking_data(self.dataset[hf_subset][split]["positive"])
                )
                negative.extend(
                    transform_reranking_data(self.dataset[hf_subset][split]["negative"])
                )
        else:
            query = self.dataset[split]["query"]
            positive = transform_reranking_data(self.dataset[split]["positive"])
            negative = transform_reranking_data(self.dataset[split]["negative"])

        len_query = [len(q) for q in query]
        total_len_query = sum(len_query)
        len_positive = [len(p) for p in positive]
        total_len_positive = sum(len_positive)
        len_negative = [len(n) for n in negative]
        total_len_negative = sum(len_negative)
        return RerankingDescriptiveStatistics(
            num_samples=len(query),
            number_of_characters=total_len_query
            + total_len_positive
            + total_len_negative,
            num_positive=len(positive),
            num_negative=len(negative),
            min_query_length=min(len_query),
            avg_query_length=total_len_query / len(query),
            max_query_length=max(len_query),
            unique_query=len(set(query)),
            min_positive_length=min(len_positive),
            avg_positive_length=total_len_positive / len(positive),
            max_positive_length=max(len_positive),
            unique_positive=len(set(positive)),
            min_negative_length=min(len_negative),
            avg_negative_length=total_len_negative / len(negative),
            max_negative_length=max(len_negative),
            unique_negative=len(set(negative)),
        )


def transform_reranking_data(data: list[list[str]] | list[str]) -> list[str]:
    """Transforms a list of lists of strings into a list of strings"""
    if isinstance(data[0], str):
        return data
    return [item for sublist in data for item in sublist]
