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

        hf_subsets = list(self.dataset) if self.is_multilingual else ["default"]

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

    def get_pairs(self, parallel: bool) -> list[tuple[str, str]]:
        pairs = [("sentence1", "sentence2")]
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

    def process_split(self, split: str, lang: str | None = None) -> dict[str, float]:
        pairs_cols = self.get_pairs(self.parallel_subsets)
        if lang:
            if self.parallel_subsets:
                sent_1, sent_2 = lang.split("-")
                sentence1 = self.dataset[split][sent_1]
                sentence2 = self.dataset[split][sent_2]
            else:
                sent_1, sent_2 = pairs_cols[0]
                sentence1 = self.dataset[lang][split][sent_1]
                sentence2 = self.dataset[lang][split][sent_2]
        else:
            sent_1, sent_2 = pairs_cols[0]
            sentence1 = self.dataset[split][sent_1]
            sentence2 = self.dataset[split][sent_2]
        total_s1_len = sum([len(s1) for s1 in sentence1])
        total_s2_len = sum([len(s2) for s2 in sentence2])

        return {
            "average_sentence1_length": total_s1_len / len(sentence1),
            "average_sentence2_length": total_s2_len / len(sentence2),
            "num_sentence1": len(sentence1),
            "num_sentence2": len(sentence2),
        }
