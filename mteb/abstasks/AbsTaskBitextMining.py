from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset, DatasetDict

from mteb._evaluators import BitextMiningEvaluator
from mteb.models import Encoder
from mteb.types import HFSubset, ScoresDict
from mteb.types.statistics import DescriptiveStatistics, TextStatistics

from ._statistics_calculation import calculate_text_statistics
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class BitextDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Bitext

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of duplicate pairs

        sentence1_statistics: Statistics for sentence1
        sentence2_statistics: Statistics for sentence2
    """

    num_samples: int
    number_of_characters: int
    unique_pairs: int

    sentence1_statistics: TextStatistics
    sentence2_statistics: TextStatistics


class AbsTaskBitextMining(AbsTask):
    """Abstract class for BitextMining tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        id: str
        sentence1: str
        sentence2: str
    """

    parallel_subsets = False
    abstask_prompt = "Retrieve parallel sentences."

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        hf_subsets = self.hf_subsets

        # If subsets_to_run is specified, filter the hf_subsets accordingly
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        scores = {}
        if self.parallel_subsets:
            scores = self._evaluate_subset(
                model,
                self.dataset[split],  # type: ignore
                parallel=True,
                hf_split=split,
                hf_subset="parallel",
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
        else:
            for hf_subset in hf_subsets:
                logger.info(
                    f"Task: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
                )

                if hf_subset not in self.dataset and hf_subset == "default":
                    data_split = self.dataset[split]
                else:
                    data_split = self.dataset[hf_subset][split]
                scores[hf_subset] = self._evaluate_subset(
                    model,
                    data_split,
                    hf_split=split,
                    hf_subset=hf_subset,
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
        hf_split: str,
        hf_subset: str,
        parallel: bool = False,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ScoresDict:
        pairs = self.get_pairs(parallel)

        evaluator = BitextMiningEvaluator(
            data_split,
            task_metadata=self.metadata,
            pair_columns=pairs,  # type: ignore
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        metrics = evaluator(model, encode_kwargs=encode_kwargs)
        if parallel:
            for v in metrics.values():
                self._add_main_score(v)
        else:
            self._add_main_score(metrics)
        return metrics

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> BitextDescriptiveStatistics:
        pairs_cols = self.get_pairs(self.parallel_subsets)
        if hf_subset:
            if self.parallel_subsets:
                sent_1, sent_2 = hf_subset.split("-")
                sentence1 = self.dataset[split][sent_1]
                sentence2 = self.dataset[split][sent_2]
            else:
                sent_1, sent_2 = pairs_cols[0]
                sentence1 = self.dataset[hf_subset][split][sent_1]
                sentence2 = self.dataset[hf_subset][split][sent_2]
        elif compute_overall:
            sentence1, sentence2 = [], []
            if self.parallel_subsets:
                for hf_subset in self.metadata.eval_langs:
                    sent_1, sent_2 = hf_subset.split("-")
                    sentence1.extend(self.dataset[split][sent_1])
                    sentence2.extend(self.dataset[split][sent_2])
            else:
                sent_1, sent_2 = pairs_cols[0]
                for hf_subset in self.metadata.eval_langs:
                    sentence1.extend(self.dataset[hf_subset][split][sent_1])
                    sentence2.extend(self.dataset[hf_subset][split][sent_2])
        else:
            sent_1, sent_2 = pairs_cols[0]
            sentence1 = self.dataset[split][sent_1]
            sentence2 = self.dataset[split][sent_2]

        text1_statistics = calculate_text_statistics(sentence1)
        text2_statistics = calculate_text_statistics(sentence2)
        unique_pairs = len(set(zip(sentence1, sentence2)))

        return BitextDescriptiveStatistics(
            num_samples=len(sentence1),
            number_of_characters=(
                text1_statistics["total_text_length"]
                + text2_statistics["total_text_length"]
            ),
            unique_pairs=unique_pairs,
            sentence1_statistics=text1_statistics,
            sentence2_statistics=text2_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        if self.metadata.is_multilingual:
            for config in self.metadata.eval_langs:
                logger.info(f"Converting {config} of {self.metadata.name}")

                sentences = {}
                if self.parallel_subsets:
                    # If there are parallel subsets, process them
                    for split in self.dataset:
                        sent_1, sent_2 = config.split("-")
                        sentences[split] = Dataset.from_dict(
                            {
                                "sentence1": self.dataset[split][sent_1],
                                "sentence2": self.dataset[split][sent_2],
                            }
                        )
                else:
                    # Handle the non-parallel subset case
                    sent_1, sent_2 = self.get_pairs(self.parallel_subsets)[0]
                    for split in self.dataset[config]:
                        sentences[split] = Dataset.from_dict(
                            {
                                "sentence1": self.dataset[config][split][sent_1],
                                "sentence2": self.dataset[config][split][sent_2],
                            }
                        )
                sentences = DatasetDict(sentences)
                sentences.push_to_hub(
                    repo_name, config, commit_message=f"Add {config} subset"
                )
        else:
            sentences = {}
            for split in self.dataset:
                sent_1, sent_2 = self.get_pairs(self.parallel_subsets)[0]
                sentences[split] = Dataset.from_dict(
                    {
                        "sentence1": self.dataset[split][sent_1],
                        "sentence2": self.dataset[split][sent_2],
                    }
                )
            sentences = DatasetDict(sentences)
            sentences.push_to_hub(repo_name)
