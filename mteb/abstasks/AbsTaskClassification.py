from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
import tqdm

from mteb.encoder_interface import Encoder

from ..evaluation.evaluators import (
    kNNClassificationEvaluator,
    kNNClassificationEvaluatorPytorch,
    logRegClassificationEvaluator,
)
from ..load_results.task_results import HFSubset, ScoresDict
from .AbsTask import AbsTask, DescriptiveStatistics

logger = logging.getLogger(__name__)


class ClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Classification

    Attributes:
      num_samples: number of samples in the dataset.
      average_text_length: Average length of text
      unique_labels: Number of unique labels
      labels: dict of label frequencies
    """

    num_samples: int
    average_text_length: float
    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskClassification(AbsTask):
    """Abstract class for kNN classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It
    must contain the following columns:
        text: str
        label: int
    """

    def __init__(
        self,
        method: str = "logReg",
        n_experiments: int | None = None,
        samples_per_label: int | None = None,
        k: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.method = method

        # Bootstrap parameters
        self.n_experiments: int = (  # type: ignore
            n_experiments
            if n_experiments is not None
            else self.metadata_dict.get("n_experiments", 10)
        )
        self.samples_per_label: int = (  # type: ignore
            samples_per_label
            if samples_per_label is not None
            else self.metadata_dict.get("samples_per_label", 8)
        )

        # kNN parameters
        self.k = k

    def _add_main_score(self, scores: dict[HFSubset, ScoresDict]) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def evaluate(
        self,
        model,
        eval_split: str = "test",
        train_split: str = "train",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        scores = {}
        hf_subsets = list(self.dataset) if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(
                f"\nTask: {self.metadata.name}, split: {eval_split}, subset: {hf_subset}. Running..."
            )

            if hf_subset not in self.dataset and hf_subset == "default":
                ds = self.dataset
            else:
                ds = self.dataset[hf_subset]
            scores[hf_subset] = self._evaluate_subset(
                model,
                ds,
                eval_split,
                train_split,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
            self._add_main_score(scores[hf_subset])

        return scores

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset,
        eval_split: str = "test",
        train_split: str = "train",
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]
        params = {"k": self.k}
        params.update(kwargs)

        scores = []
        test_cache, idxs = (
            None,
            None,
        )  # we store idxs to make the shuffling reproducible
        for i in range(self.n_experiments):
            logger.info(
                "=" * 10 + f" Experiment {i+1}/{self.n_experiments} " + "=" * 10
            )
            # Bootstrap `self.samples_per_label` samples per label for each split
            X_sampled, y_sampled, idxs = self._undersample_data(
                train_split["text"],  # type: ignore
                train_split["label"],  # type: ignore
                self.samples_per_label,
                idxs,
            )

            if self.method == "kNN":
                evaluator = kNNClassificationEvaluator(
                    X_sampled,
                    y_sampled,
                    eval_split["text"],  # type: ignore
                    eval_split["label"],  # type: ignore
                    task_name=self.metadata.name,
                    encode_kwargs=encode_kwargs,
                    **params,
                )
            elif self.method == "kNN-pytorch":
                evaluator = kNNClassificationEvaluatorPytorch(
                    X_sampled,
                    y_sampled,
                    eval_split["text"],  # type: ignore
                    eval_split["label"],  # type: ignore
                    task_name=self.metadata.name,
                    encode_kwargs=encode_kwargs,
                    **params,
                )
            elif self.method == "logReg":
                evaluator = logRegClassificationEvaluator(
                    X_sampled,
                    y_sampled,
                    eval_split["text"],  # type: ignore
                    eval_split["label"],  # type: ignore
                    task_name=self.metadata.name,
                    encode_kwargs=encode_kwargs,
                    **params,
                )
            else:
                raise ValueError(f"Method {self.method} not supported")

            scores_exp, test_cache = evaluator(model, test_cache=test_cache)
            scores.append(scores_exp)

        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in scores]) for k in scores[0].keys()
        }
        avg_scores["scores_per_experiment"] = scores
        return avg_scores

    def _undersample_data(self, X, y, samples_per_label: int, idxs=None):
        """Undersample data to have samples_per_label samples of each label"""
        X_sampled = []
        y_sampled = []
        if idxs is None:
            idxs = np.arange(len(y))
        np.random.shuffle(idxs)
        label_counter = defaultdict(int)
        for i in idxs:
            if label_counter[y[i]] < samples_per_label:
                X_sampled.append(X[i])
                y_sampled.append(y[i])
                label_counter[y[i]] += 1
        return X_sampled, y_sampled, idxs

    def calculate_metadata_metrics(
        self,
    ) -> dict[
        str,
        ClassificationDescriptiveStatistics
        | dict[str, ClassificationDescriptiveStatistics],
    ]:
        self.load_data()

        # same function from parent class, but added explicitly train to splits

        all_details = {}
        pbar_split = tqdm.tqdm(
            self.metadata.eval_splits + ["train"], desc="Processing Splits..."
        )
        for split in pbar_split:
            pbar_split.set_postfix_str(f"Split: {split}")
            logger.info(f"Processing metadata for split {split}")
            if self.is_multilingual:
                all_details[split] = self._calculate_metrics_from_split(
                    split, compute_overall=True
                )
                all_details[split]["hf_subset_descriptive_stats"] = {}

                pbar_subset = tqdm.tqdm(
                    self.metadata.eval_langs, desc="Processing Languages..."
                )
                for hf_subset in pbar_subset:
                    pbar_subset.set_postfix_str(f"Language: {hf_subset}")
                    logger.info(f"Processing metadata for language {hf_subset}")
                    split_details = self._calculate_metrics_from_split(split, hf_subset)
                    all_details[split][hf_subset] = split_details
            else:
                split_details = self._calculate_metrics_from_split(split)
                all_details[split] = split_details

        return all_details

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClassificationDescriptiveStatistics:
        if hf_subset:
            text = self.dataset[hf_subset][split]["text"]
            label = self.dataset[hf_subset][split]["label"]
        elif compute_overall:
            text = []
            label = []
            for hf_subset in self.metadata.eval_langs:
                text.extend(self.dataset[hf_subset][split]["text"])
                label.extend(self.dataset[hf_subset][split]["label"])
        else:
            text = self.dataset[split]["text"]
            label = self.dataset[split]["label"]

        total_text_len = sum([len(t) for t in text])
        label_count = Counter(label)
        return ClassificationDescriptiveStatistics(
            num_samples=len(text),
            average_text_length=total_text_len / len(text),
            unique_labels=len(label_count),
            labels={
                str(label): {"count": count} for label, count in label_count.items()
            },
        )
