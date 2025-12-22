import logging
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset

from mteb._evaluators import SummarizationEvaluator
from mteb._evaluators.text.summarization_evaluator import SummarizationMetrics
from mteb.abstasks._statistics_calculation import (
    calculate_score_statistics,
    calculate_text_statistics,
)
from mteb.abstasks.abstask import AbsTask
from mteb.models import EncoderProtocol, MTEBModels
from mteb.types.statistics import (
    ScoreStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

logger = logging.getLogger(__name__)


class SummarizationDescriptiveStatistics(SplitDescriptiveStatistics):
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

    Attributes:
        dataset: HuggingFace dataset containing the data for the task. Should have columns:
            - text: The original text to be summarized.
            - human_summaries: A list of human-written summaries for the text.
            - machine_summaries: A list of machine-generated summaries for the text.
            - relevance: A list of relevance scores (integers) corresponding to each machine summary, indicating how relevant each summary is to the original text.
        min_score: Minimum possible relevance score (inclusive).
        max_score: Maximum possible relevance score (inclusive).
        human_summaries_column_name: Name of the column containing human summaries.
        machine_summaries_column_name: Name of the column containing machine summaries.
        text_column_name: Name of the column containing the original text.
        relevancy_column_name: Name of the column containing relevance scores.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    min_score: int
    max_score: int

    abstask_prompt = (
        "Given a news summary, retrieve other semantically similar summaries."
    )
    # SummEval has DeprecatedSummarizationEvaluator
    evaluator = SummarizationEvaluator
    text_column_name: str = "text"
    human_summaries_column_name: str = "human_summaries"
    machine_summaries_column_name: str = "machine_summaries"
    relevancy_column_name: str = "relevance"

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        prediction_folder: Path | None = None,
        **kwargs,
    ) -> SummarizationMetrics:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        normalized_scores = [
            (
                (np.array(x) - self.min_score) / (self.max_score - self.min_score)
            ).tolist()
            for x in data_split[self.relevancy_column_name]
        ]
        evaluator = self.evaluator(
            machine_summaries=data_split[self.machine_summaries_column_name],
            human_summaries=data_split[self.human_summaries_column_name],
            texts=data_split[self.text_column_name],
            gold_scores=normalized_scores,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        if prediction_folder:
            self._save_task_predictions(
                scores,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )
        return evaluator._calculate_metrics(scores)

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> SummarizationDescriptiveStatistics:
        if hf_subset:
            text = self.dataset[hf_subset][split][self.text_column_name]
            human_summaries = self.dataset[hf_subset][split][
                self.human_summaries_column_name
            ]
            machine_summaries = self.dataset[hf_subset][split][
                self.machine_summaries_column_name
            ]
            relevance = self.dataset[hf_subset][split][self.relevancy_column_name]
        elif compute_overall:
            text = []
            human_summaries = []
            machine_summaries = []
            relevance = []

            for hf_subset in self.metadata.eval_langs:
                text.extend(self.dataset[hf_subset][split][self.text_column_name])
                human_summaries.extend(
                    self.dataset[hf_subset][split][self.human_summaries_column_name]
                )
                machine_summaries.extend(
                    self.dataset[hf_subset][split][self.machine_summaries_column_name]
                )
                relevance.extend(
                    self.dataset[hf_subset][split][self.relevancy_column_name]
                )
        else:
            text = self.dataset[split][self.text_column_name]
            human_summaries = self.dataset[split][self.human_summaries_column_name]
            machine_summaries = self.dataset[split][self.machine_summaries_column_name]
            relevance = self.dataset[split][self.relevancy_column_name]

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
