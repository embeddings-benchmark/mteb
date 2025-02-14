from __future__ import annotations

import logging
import warnings
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from mteb.encoder_interface import Encoder

from ..evaluation.evaluators import (
    kNNClassificationEvaluator,
    kNNClassificationEvaluatorPytorch,
    logRegClassificationEvaluator,
)
from ..load_results.task_results import HFSubset, ScoresDict
from .AbsTask import AbsTask
from .TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class ClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Classification

    Attributes:
      num_samples: number of samples in the dataset.
      number_of_characters: Total number of symbols in the dataset.
      num_texts_in_train: Number of texts in the train split

      min_text_length: Minimum length of text
      average_text_length: Average length of text
      max_text_length: Maximum length of text
      unique_text: Number of unique texts

      unique_labels: Number of unique labels
      labels: dict of label frequencies
    """

    num_samples: int
    number_of_characters: int
    num_texts_in_train: int | None

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_text: int

    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskClassification(AbsTask):
    """Abstract class for kNN classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It
    must contain the following columns:
        text: str
        label: int

    Attributes:
       samples_per_label: Number of samples to use pr. label. These samples are embedded and a classifier is fit using the labels and samples.

    """

    abstask_prompt = "Classify user passages."
    samples_per_label: int = 8

    def __init__(
        self,
        method: str = "logReg",
        n_experiments: int | None = None,
        k: int = 3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if method != "logReg":
            warnings.warn(
                "Passing `method` to AbsTaskClassification is deprecated and will be removed in v2.0.0.",
                DeprecationWarning,
            )
        self.method = method

        if n_experiments:
            warnings.warn(
                "Passing `n_experiments` to AbsTaskClassification is deprecated and will be removed in v2.0.0.",
                DeprecationWarning,
            )

        # Bootstrap parameters
        self.n_experiments: int = (  # type: ignore
            n_experiments
            if n_experiments is not None
            else self.metadata_dict.get("n_experiments", 10)
        )

        if k != 3:
            warnings.warn(
                "Passing `k` to AbsTaskClassification is deprecated and will be removed in v2.0.0.",
                DeprecationWarning,
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
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        if train_split != "train":
            warnings.warn(
                "Passing `train_split` to AbsTaskClassification.evaluate is deprecated and will be removed in v2.0.0.",
                DeprecationWarning,
            )

        scores = {}
        hf_subsets = list(self.dataset) if self.is_multilingual else ["default"]
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

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
                "=" * 10 + f" Experiment {i + 1}/{self.n_experiments} " + "=" * 10
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

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClassificationDescriptiveStatistics:
        train_text = []
        if hf_subset:
            text = self.dataset[hf_subset][split]["text"]
            label = self.dataset[hf_subset][split]["label"]
            if split != "train":
                train_text = self.dataset[hf_subset]["train"]["text"]
        elif compute_overall:
            text = []
            label = []
            for hf_subset in self.metadata.eval_langs:
                text.extend(self.dataset[hf_subset][split]["text"])
                label.extend(self.dataset[hf_subset][split]["label"])
                if split != "train":
                    train_text.extend(self.dataset[hf_subset]["train"]["text"])
        else:
            text = self.dataset[split]["text"]
            label = self.dataset[split]["label"]
            if split != "train":
                train_text = self.dataset["train"]["text"]

        text_len = [len(t) for t in text]
        total_text_len = sum(text_len)
        label_count = Counter(label)
        num_texts_in_train = (
            len(set(text) & set(train_text)) if split != "train" else None
        )
        return ClassificationDescriptiveStatistics(
            num_samples=len(text),
            number_of_characters=total_text_len,
            num_texts_in_train=num_texts_in_train,
            min_text_length=min(text_len),
            average_text_length=total_text_len / len(text),
            max_text_length=max(text_len),
            unique_text=len(set(text)),
            unique_labels=len(label_count),
            labels={
                str(label): {"count": count} for label, count in label_count.items()
            },
        )
