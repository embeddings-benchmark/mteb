from __future__ import annotations

import logging
from typing import Any

from ..evaluation.evaluators import STSEvaluator
from ..load_results.task_results import ScoresDict
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class STSDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for STS

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.

        min_sentence1_length: Minimum length of sentence1
        average_sentence1_len: Average length of sentence1
        max_sentence1_length: Maximum length of sentence1

        min_sentence2_length: Minimum length of sentence2
        average_sentence2_len: Average length of sentence2
        max_sentence2_length: Maximum length of sentence2

        min_score: Minimum score
        avg_score: Average score
        max_score: Maximum score
    """

    num_samples: int
    number_of_characters: int

    min_sentence1_length: int
    average_sentence1_len: float
    max_sentence1_length: int
    unique_sentence1: int

    min_sentence2_length: int
    average_sentence2_len: float
    max_sentence2_length: int
    unique_sentence2: int

    min_score: float
    avg_score: float
    max_score: float


class AbsTaskSTS(AbsTask):
    """Abstract class for STS experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns::
        sentence1: str
        sentence2: str
        score: float
    """

    abstask_prompt = "Retrieve semantically similar text."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def min_score(self) -> int:
        return self.metadata_dict["min_score"]

    @property
    def max_score(self) -> int:
        return self.metadata_dict["max_score"]

    def _evaluate_subset(
        self, model, data_split, *, encode_kwargs: dict[str, Any] = {}, **kwargs
    ) -> ScoresDict:
        def normalize(x):
            return (x - self.min_score) / (self.max_score - self.min_score)

        normalized_scores = list(map(normalize, data_split["score"]))
        evaluator = STSEvaluator(
            data_split["sentence1"],
            data_split["sentence2"],
            normalized_scores,
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
    ) -> STSDescriptiveStatistics:
        if hf_subset:
            sentence1 = self.dataset[hf_subset][split]["sentence1"]
            sentence2 = self.dataset[hf_subset][split]["sentence2"]
            score = self.dataset[hf_subset][split]["score"]
        elif compute_overall:
            sentence1 = []
            sentence2 = []
            score = []
            for hf_subset in self.metadata.eval_langs:
                sentence1.extend(self.dataset[hf_subset][split]["sentence1"])
                sentence2.extend(self.dataset[hf_subset][split]["sentence2"])
                score.extend(self.dataset[hf_subset][split]["score"])
        else:
            sentence1 = self.dataset[split]["sentence1"]
            sentence2 = self.dataset[split]["sentence2"]
            score = self.dataset[split]["score"]

        sentence1_len = [len(s) for s in sentence1]
        sentence2_len = [len(s) for s in sentence2]
        total_sentence1_len = sum(sentence1_len)
        total_sentence2_len = sum(sentence2_len)
        avg_score = sum(score) / len(score)
        return STSDescriptiveStatistics(
            num_samples=len(sentence1),
            number_of_characters=total_sentence1_len + total_sentence2_len,
            min_sentence1_length=min(sentence1_len),
            average_sentence1_len=total_sentence1_len / len(sentence1),
            max_sentence1_length=max(sentence1_len),
            unique_sentence1=len(set(sentence1)),
            min_sentence2_length=min(sentence2_len),
            average_sentence2_len=total_sentence2_len / len(sentence2),
            max_sentence2_length=max(sentence2_len),
            unique_sentence2=len(set(sentence2)),
            min_score=min(score),
            avg_score=avg_score,
            max_score=max(score),
        )
