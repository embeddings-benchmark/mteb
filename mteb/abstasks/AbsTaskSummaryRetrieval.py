from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from mteb.encoder_interface import Encoder

from ..evaluation.evaluators import SummaryRetrievalEvaluator
from ..load_results.task_results import HFSubset, ScoresDict
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class SummaryRetrievalDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Summary Retrieval

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of duplicate pairs

        min_text_length: Minimum length of text
        average_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_text: Number of duplicates in text

        min_summary_length: Minimum length of summary
        average_summary_length: Average length of summary
        max_summary_length: Maximum length of summary
    """

    num_samples: int
    number_of_characters: int
    unique_pairs: int

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_text: int

    min_summary_length: int
    average_summary_length: float
    max_summary_length: int
    unique_summary: int


class AbsTaskSummaryRetrieval(AbsTask):
    """Abstract class for SummaryRetrieval tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        id: str
        text: str
        summary: str
    """

    parallel_subsets = False
    # abstask_prompt = ""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        hf_subsets = list(self.dataset) if self.is_multilingual else ["default"]

        # If subsets_to_run is specified, filter the hf_subsets accordingly
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        scores = {}
        if self.parallel_subsets:
            scores = self._evaluate_subset(
                model,
                self.dataset[split],  # type: ignore
                parallel=True,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
        else:
            for hf_subet in hf_subsets:
                logger.info(
                    f"\nTask: {self.metadata.name}, split: {split}, subset: {hf_subet}. Running..."
                )

                if hf_subet not in self.dataset and hf_subet == "default":
                    data_split = self.dataset[split]
                else:
                    data_split = self.dataset[hf_subet][split]
                scores[hf_subet] = self._evaluate_subset(
                    model,
                    data_split,  # type: ignore
                    subsets=["text", "summary"],
                    encode_kwargs=encode_kwargs,
                    **kwargs,
                )

        return scores

    def get_pairs(self, parallel: bool) -> list[tuple[str, str]]:
        pairs = [("text", "summary")]
        if parallel:
            pairs = [langpair.split("-") for langpair in self.hf_subsets]
        return pairs

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        *,
        parallel: bool = False,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        pairs = self.get_pairs(parallel)

        evaluator = SummaryRetrievalEvaluator(
            data_split,
            task_name=self.metadata.name,
            pair_columns=pairs,  # type: ignore
            **kwargs,
        )
        metrics = evaluator(model, encode_kwargs=encode_kwargs)
        if parallel:
            for v in metrics.values():
                self._add_main_score(v)
        else:
            self._add_main_score(metrics)
        return metrics

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> SummaryRetrievalDescriptiveStatistics:
        pairs_cols = self.get_pairs(self.parallel_subsets)
        if hf_subset:
            if self.parallel_subsets:
                sent_1, sent_2 = hf_subset.split("-")
                text = self.dataset[split][sent_1]
                summary = self.dataset[split][sent_2]
            else:
                sent_1, sent_2 = pairs_cols[0]
                text = self.dataset[hf_subset][split][sent_1]
                summary = self.dataset[hf_subset][split][sent_2]
        elif compute_overall:
            text = []
            summary = []
            if self.parallel_subsets:
                for hf_subset in self.metadata.eval_langs:
                    sent_1, sent_2 = hf_subset.split("-")
                    text.extend(self.dataset[split][sent_1])
                    summary.extend(self.dataset[split][sent_2])
            else:
                sent_1, sent_2 = pairs_cols[0]
                for hf_subset in self.metadata.eval_langs:
                    text.extend(self.dataset[hf_subset][split][sent_1])
                    summary.extend(self.dataset[hf_subset][split][sent_2])
        else:
            sent_1, sent_2 = pairs_cols[0]
            text = self.dataset[split][sent_1]
            summary = self.dataset[split][sent_2]
        s1_len = [len(s1) for s1 in text]
        s2_len = [len(s2) for s2 in summary]
        total_s1_len = sum(s1_len)
        total_s2_len = sum(s2_len)

        unique_pairs = len(set(zip(text, summary)))
        unique_text = len(set(text))
        unique_summary = len(set(summary))
        return SummaryRetrievalDescriptiveStatistics(
            num_samples=len(text),
            number_of_characters=total_s1_len + total_s2_len,
            unique_pairs=unique_pairs,
            min_text_length=min(s1_len),
            average_text_length=sum(s1_len) / len(text),
            max_text_length=max(s1_len),
            unique_text=unique_text,
            min_summary_length=min(s2_len),
            average_summary_length=total_s2_len / len(summary),
            max_summary_length=max(s2_len),
            unique_summary=unique_summary,
        )
