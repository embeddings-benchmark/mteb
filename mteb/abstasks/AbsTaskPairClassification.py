from __future__ import annotations

import logging
from collections import defaultdict

from datasets import Dataset

from mteb._evaluators import PairClassificationEvaluator
from mteb.types import ScoresDict
from mteb.types.statistics import (
    LabelStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

from ..models.models_protocols import Encoder
from ._statistics_calculation import (
    calculate_label_statistics,
    calculate_text_statistics,
)
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class PairClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for PairClassification

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of unique pairs

        text1_statistics: Statistics for sentence1
        text2_statistics: Statistics for sentence2
        labels_statistics: Statistics for labels
    """

    num_samples: int
    number_of_characters: int
    unique_pairs: int

    text1_statistics: TextStatistics
    text2_statistics: TextStatistics
    labels_statistics: LabelStatistics


class AbsTaskPairClassification(AbsTask):
    """Abstract class for PairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        sentence1: list[str]
        sentence2: list[str]
        labels: list[int]
    """

    abstask_prompt = "Retrieve text that are semantically similar to the given text."

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, str] = {},
        **kwargs,
    ) -> ScoresDict:
        data_split = data_split[0] if len(data_split) == 1 else data_split
        logging.getLogger(
            "sentence_transformers.evaluation.PairClassificationEvaluator"
        ).setLevel(logging.WARN)
        evaluator = PairClassificationEvaluator(
            data_split["sentence1"],
            data_split["sentence2"],
            data_split["labels"],
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            **kwargs,
        )
        scores = evaluator.compute_metrics(model, encode_kwargs=encode_kwargs)

        self._add_main_score(scores)
        return scores

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> PairClassificationDescriptiveStatistics:
        if hf_subset:
            dataset = self.dataset[hf_subset][split]
        elif compute_overall:
            dataset = defaultdict(list)
            for hf_subset in self.metadata.eval_langs:
                cur_dataset = self.dataset[hf_subset][split]
                if isinstance(cur_dataset, list):
                    cur_dataset = cur_dataset[0]
                for key, value in cur_dataset.items():
                    dataset[key].extend(value[0] if len(value) == 1 else value)
        else:
            dataset = self.dataset[split]

        if isinstance(dataset, list):
            dataset = dataset[0]

        sentence1 = (
            dataset["sentence1"][0]
            if len(dataset["sentence1"]) == 1
            else dataset["sentence1"]
        )
        sentence2 = (
            dataset["sentence2"][0]
            if len(dataset["sentence2"]) == 1
            else dataset["sentence2"]
        )
        labels = (
            dataset["labels"][0] if len(dataset["labels"]) == 1 else dataset["labels"]
        )

        text1_statistics = calculate_text_statistics(sentence1)
        text2_statistics = calculate_text_statistics(sentence2)
        return PairClassificationDescriptiveStatistics(
            num_samples=len(sentence1),
            number_of_characters=(
                text1_statistics["total_text_length"]
                + text2_statistics["total_text_length"]
            ),
            unique_pairs=len(set(zip(sentence1, sentence2))),
            text1_statistics=text1_statistics,
            text2_statistics=text2_statistics,
            labels_statistics=calculate_label_statistics(labels),
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        # previously pair classification datasets were stored in a single row
        if self.metadata.is_multilingual:
            for subset in self.dataset:
                for split in self.dataset[subset]:
                    if len(self.dataset[subset][split]) == 1:
                        self.dataset[subset][split] = self.dataset[subset][split][0]
        else:
            for split in self.dataset:
                if len(self.dataset[split]) == 1:
                    self.dataset[split] = self.dataset[split][0]
        self._upload_dataset_to_hub(repo_name, ["sentence1", "sentence2", "labels"])
