from __future__ import annotations

import logging
from typing import Any

from ..evaluation.evaluators import STSEvaluator
from ..load_results.mteb_results import ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskSTS(AbsTask):
    """Abstract class for STS experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns::
        sentence1: str
        sentence2: str
        score: float
    """

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

    def process_split(self, split: str, lang: str | None = None) -> dict[str, float]:
        """sentence1: str
        sentence2: str
        score: float
        """
        if lang:
            sentence1 = self.dataset[lang][split]["sentence1"]
            sentence2 = self.dataset[lang][split]["sentence2"]
            score = self.dataset[lang][split]["score"]
        else:
            sentence1 = self.dataset[split]["sentence1"]
            sentence2 = self.dataset[split]["sentence2"]
            score = self.dataset[split]["score"]

        total_sentence1_len = sum([len(s) for s in sentence1])
        total_sentence2_len = sum([len(s) for s in sentence2])
        avg_score = sum(score) / len(score)
        return {
            "num_sentence1": len(sentence1),
            "num_sentence2": len(sentence2),
            "average_sentence1_len": total_sentence1_len / len(sentence1),
            "average_sentence2_len": total_sentence2_len / len(sentence2),
            "avg_score": avg_score,
        }
