from __future__ import annotations

import logging
from collections import Counter, defaultdict

from datasets import Dataset

from ..encoder_interface import Encoder
from ..evaluation.evaluators import PairClassificationEvaluator
from ..load_results.task_results import ScoresDict
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class PairClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for PairClassification

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.

        min_sentence1_length: Minimum length of sentence1
        avg_sentence1_length: Average length of sentence1
        max_sentence1_length: Maximum length of sentence1
        unique_sentence1: Number of unique sentence

        min_sentence2_length: Minimum length of sentence2
        avg_sentence2_length: Average length of sentence2
        max_sentence2_length: Maximum length of sentence2
        unique_sentence2: Number of unique sentence

        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int
    number_of_characters: int

    min_sentence1_length: int
    avg_sentence1_length: float
    max_sentence1_length: int
    unique_sentence1: int

    min_sentence2_length: int
    avg_sentence2_length: float
    max_sentence2_length: int
    unique_sentence2: int

    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskPairClassification(AbsTask):
    """Abstract class for PairClassificationTasks
    The similarity is computed between pairs and the results are ranked. Average precision
    is computed to measure how well the methods can be used for pairwise pair classification.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sentence1: list[str]
        sentence2: list[str]
        labels: list[int]
    """

    abstask_prompt = "Retrieve text that are semantically similar to the given text."

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, str] = {},
        **kwargs,
    ) -> ScoresDict:
        data_split = dataset[0]
        logging.getLogger(
            "sentence_transformers.evaluation.PairClassificationEvaluator"
        ).setLevel(logging.WARN)
        evaluator = PairClassificationEvaluator(
            data_split["sentence1"],
            data_split["sentence2"],
            data_split["labels"],
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator.compute_metrics(model, encode_kwargs=encode_kwargs)

        self._add_main_score(scores)
        return scores

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> PairClassificationDescriptiveStatistics:
        if hf_subset:
            dataset = self.dataset[hf_subset][split]
            if isinstance(dataset, list):
                dataset = dataset[0]
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

        sentence1_len = [len(sentence) for sentence in sentence1]
        total_sentence1_len = sum(sentence1_len)
        sentence2_len = [len(sentence) for sentence in sentence2]
        total_sentence2_len = sum(sentence2_len)
        label_count = Counter(labels)
        return PairClassificationDescriptiveStatistics(
            num_samples=len(sentence1),
            number_of_characters=total_sentence1_len + total_sentence2_len,
            min_sentence1_length=min(sentence1_len),
            avg_sentence1_length=total_sentence1_len / len(sentence1),
            max_sentence1_length=max(sentence1_len),
            unique_sentence1=len(set(sentence1)),
            min_sentence2_length=min(sentence2_len),
            avg_sentence2_length=total_sentence2_len / len(sentence2),
            max_sentence2_length=max(sentence2_len),
            unique_sentence2=len(set(sentence2)),
            unique_labels=len(set(labels)),
            labels={
                str(label): {"count": count} for label, count in label_count.items()
            },
        )
