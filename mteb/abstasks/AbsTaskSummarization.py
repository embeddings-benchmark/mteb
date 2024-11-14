from __future__ import annotations

import logging
from typing import Any

import numpy as np

from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import ScoresDict

from ..evaluation.evaluators import SummarizationEvaluator
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class SummarizationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Summarization

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.

        min_text_length: Minimum length of text
        avg_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_texts: Number of unique texts

        min_human_summaries_length: Minimum length of human summaries
        avg_human_summaries_length: Average length of human summaries
        max_human_summaries_length: Maximum length of human summaries
        unique_human_summaries: Number of unique human summaries

        min_machine_summaries_length: Minimum length of machine summaries
        avg_machine_summaries_length: Average length of machine summaries
        max_machine_summaries_length: Maximum length of machine summaries
        unique_machine_summaries: Number of unique machine summaries

        min_relevance: Minimum relevance score
        avg_relevance: Average relevance score
        max_relevance: Maximum relevance score
    """

    num_samples: int
    number_of_characters: int

    min_text_length: int
    avg_text_length: float
    max_text_length: int
    unique_texts: int

    min_human_summaries_length: int
    avg_human_summaries_length: float
    max_human_summaries_length: int
    unique_human_summaries: int

    min_machine_summaries_length: int
    avg_machine_summaries_length: float
    max_machine_summaries_length: int
    unique_machine_summaries: int

    min_relevance: float
    avg_relevance: float
    max_relevance: float


class AbsTaskSummarization(AbsTask):
    """Abstract class for summarization experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        text: str
        human_summaries: list[str]
        machine_summaries: list[str]
        relevance: list[float] (the score of the machine generated summaries)
    """

    evalutor = SummarizationEvaluator
    abstask_prompt = (
        "Given a news summary, retrieve other semantically similar summaries."
    )

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

        all_human_summaries = []
        for s in human_summaries:
            all_human_summaries.extend(s)

        all_machine_summaries = []
        for s in machine_summaries:
            all_machine_summaries.extend(s)

        text_len = [len(t) for t in text]
        total_text_len = sum(text_len)
        human_summaries_len = [len(s) for s in human_summaries]
        total_human_summaries_len = sum(human_summaries_len)
        machine_summaries_len = [len(s) for s in machine_summaries]
        total_machine_summaries_len = sum(machine_summaries_len)
        total_relevance = sum(sum(x) / len(x) for x in relevance)
        return SummarizationDescriptiveStatistics(
            num_samples=len(text),
            number_of_characters=total_text_len
            + total_human_summaries_len
            + total_machine_summaries_len,
            min_text_length=min(text_len),
            avg_text_length=total_text_len / len(text),
            max_text_length=max(text_len),
            unique_texts=len(set(text)),
            min_human_summaries_length=min(human_summaries_len),
            avg_human_summaries_length=total_human_summaries_len / len(text),
            max_human_summaries_length=max(human_summaries_len),
            unique_human_summaries=len(set(all_human_summaries)),
            min_machine_summaries_length=min(machine_summaries_len),
            avg_machine_summaries_length=total_machine_summaries_len / len(text),
            max_machine_summaries_length=max(machine_summaries_len),
            unique_machine_summaries=len(set(all_machine_summaries)),
            min_relevance=min(relevance),
            avg_relevance=total_relevance / len(relevance),
            max_relevance=max(relevance),
        )
