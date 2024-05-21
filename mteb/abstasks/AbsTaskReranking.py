from __future__ import annotations

from typing import Any

from datasets import Dataset

from mteb.abstasks.MTEBResults import ScoresDict
from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode

from ..evaluation.evaluators import RerankingEvaluator
from .AbsTask import AbsTask


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
        **kwargs: Any,
    ) -> ScoresDict:
        evaluator = RerankingEvaluator(data_split, **kwargs)
        scores = evaluator(model)

        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]
