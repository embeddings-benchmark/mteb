from __future__ import annotations

import logging
from typing import Any

import numpy as np
from datasets import Dataset

from mteb.encoder_interface import Encoder
from mteb.types import ScoresDict
from mteb.types.statistics import DescriptiveStatistics, ScoreStatistics, TextStatistics

from ..evaluation.evaluators import SummarizationEvaluator
from ._statistics_calculation import (
    calculate_score_statistics,
    calculate_text_statistics,
)
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class SummarizationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Summarization

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.

        text_statistics: Statistics for the text
        human_summaries_statistics: Statistics for human summaries
        machine_summaries_statistics: Statistics for machine summaries
        score_statistics: Statistics for the relevance scoresk
    """

    num_samples: int
    number_of_characters: int

    text_statistics: TextStatistics
    human_summaries_statistics: TextStatistics
    machine_summaries_statistics: TextStatistics
    score_statistics: ScoreStatistics


class AbsTaskSummarization(AbsTask):
    """Abstract class for summarization experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        text: str
        human_summaries: list[str]
        machine_summaries: list[str]
        relevance: list[list[float]] (the score of the machine generated summaries)
    """

    min_score: int
    max_score: int

    abstask_prompt = (
        "Given a news summary, retrieve other semantically similar summaries."
    )
    # SummEval has DeprecatedSummarizationEvaluator
    evaluator = SummarizationEvaluator

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ScoresDict:
        normalized_scores = [
            (np.array(x) - self.min_score) / (self.max_score - self.min_score)
            for x in data_split["relevance"]
        ]
        evaluator = self.evaluator(
            machine_summaries=data_split["machine_summaries"],
            human_summaries=data_split["human_summaries"],
            texts=data_split["text"],
            gold_scores=normalized_scores,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(scores)
        return scores

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

        all_human_summaries = []
        for s in human_summaries:
            all_human_summaries.extend(s)

        all_machine_summaries = []
        for s in machine_summaries:
            all_machine_summaries.extend(s)

        text_statistics = calculate_text_statistics(text)
        human_summaries_statistics = calculate_text_statistics(all_human_summaries)
        machine_summaries_statistics = calculate_text_statistics(all_machine_summaries)

        relevance = [item for sublist in relevance for item in sublist]

        return SummarizationDescriptiveStatistics(
            num_samples=len(text),
            number_of_characters=(
                text_statistics["total_text_length"]
                + human_summaries_statistics["total_text_length"]
                + machine_summaries_statistics["total_text_length"]
            ),
            text_statistics=text_statistics,
            human_summaries_statistics=human_summaries_statistics,
            machine_summaries_statistics=machine_summaries_statistics,
            score_statistics=calculate_score_statistics(relevance),
        )
