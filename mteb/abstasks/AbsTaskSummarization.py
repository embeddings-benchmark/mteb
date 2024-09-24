from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import ScoresDict

from ..evaluation.evaluators import SummarizationEvaluator
from .AbsTask import AbsTask, DescriptiveStatistics

logger = logging.getLogger(__name__)


class SummarizationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Summarization

    Attributes:
        num_samples: number of samples in the dataset.
        avg_text_len: Average length of text
        avg_human_summaries_len: Average length of human summaries
        avg_machine_summaries_len: Average length of machine summaries
        avg_relevance: Average relevance score
    """

    num_samples: int
    avg_text_len: float
    avg_human_summaries_len: float
    avg_machine_summaries_len: float
    avg_relevance: float


class AbsTaskSummarization(AbsTask):
    """Abstract class for summarization experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        text: str
        human_summaries: list[str]
        machine_summaries: list[str]
        relevance: list[float] (the score of the machine generated summaries)
    """

    evalutor = SummarizationEvaluator

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self):
        return self.metadata_dict["min_score"]

    @property
    def max_score(self):
        return self.metadata_dict["max_score"]

    def _evaluate_subset(
        self, model: Encoder, data_split, *, encode_kwargs: dict[str, Any], **kwargs
    ) -> ScoresDict:
        normalized_scores = [
            (np.array(x) - self.min_score) / (self.max_score - self.min_score)
            for x in data_split["relevance"]
        ]
        evaluator = self.evalutor(
            machine_summaries=data_split["machine_summaries"],
            human_summaries=data_split["human_summaries"],
            texts=data_split["text"],
            gold_scores=normalized_scores,
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> SummarizationDescriptiveStatistics:
        if hf_subset:
            text = self.dataset[hf_subset][split]["text"]
            human_summaries = self.dataset[hf_subset][split]["human_summaries"]
            machine_summaries = self.dataset[hf_subset][split]["machine_summaries"]
            relevance = self.dataset[hf_subset][split]["relevance"]
        elif compute_overall:
            text = []
            human_summaries = []
            machine_summaries = []
            relevance = []

            for hf_subset in self.metadata.eval_langs:
                text.extend(self.dataset[hf_subset][split]["text"])
                human_summaries.extend(
                    self.dataset[hf_subset][split]["human_summaries"]
                )
                machine_summaries.extend(
                    self.dataset[hf_subset][split]["machine_summaries"]
                )
                relevance.extend(self.dataset[hf_subset][split]["relevance"])
        else:
            text = self.dataset[split]["text"]
            human_summaries = self.dataset[split]["human_summaries"]
            machine_summaries = self.dataset[split]["machine_summaries"]
            relevance = self.dataset[split]["relevance"]

        total_text_len = sum(len(x) for x in text)
        total_human_summaries_len = sum(len(x) for x in human_summaries)
        total_machine_summaries_len = sum(len(x) for x in machine_summaries)
        total_relevance = sum(sum(x) / len(x) for x in relevance)
        return SummarizationDescriptiveStatistics(
            num_samples=len(text),
            avg_text_len=total_text_len / len(text),
            avg_human_summaries_len=total_human_summaries_len / len(text),
            avg_machine_summaries_len=total_machine_summaries_len / len(text),
            avg_relevance=total_relevance / len(relevance),
        )
