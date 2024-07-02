from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from mteb.encoder_interface import Encoder

from ..evaluation.evaluators import BitextMiningEvaluator
from ..load_results.mteb_results import HFSubset, ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskBitextMining(AbsTask):
    """Abstract class for BitextMining tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        id: str
        sentence1: str
        sentence2: str
    """

    parallel_subsets = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(
        self,
        model: Encoder,
        split: str,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        hf_subsets = [l for l in self.dataset] if self.is_multilingual else ["default"]

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
                    subsets=["sentence1", "sentence2"],
                    encode_kwargs=encode_kwargs,
                    **kwargs,
                )

        return scores

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        *,
        parallel: bool = False,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        pairs = [("sentence1", "sentence2")]
        if parallel:
            pairs = [langpair.split("-") for langpair in self.hf_subsets]

        evaluator = BitextMiningEvaluator(
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
