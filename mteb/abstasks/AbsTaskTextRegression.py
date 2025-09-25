from __future__ import annotations

import logging
from typing import Any

import datasets
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.TaskMetadata import DescriptiveStatistics, HFSubset
from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RegressionEvaluator import (
    LinearRegressionEvaluator,
    SklearnRegressorModel,
)
from mteb.load_results.task_results import ScoresDict

logger = logging.getLogger(__name__)


class RegressionDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Regression

    Attributes:
      num_samples: number of samples in the dataset.
      number_of_characters: Total number of symbols in the dataset.
      num_texts_in_train: Number of texts in the train split

      min_text_length: Minimum length of text
      average_text_length: Average length of text
      max_text_length: Maximum length of text
      unique_text: Number of unique texts

      min_value: Minimum of the target variable
      average_value: Average of the target variable
      max_value: Maximum of the target variable
    """

    num_samples: int
    number_of_characters: int
    num_texts_in_train: int | None

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_text: int

    min_value: float
    average_value: float
    max_value: float


class AbsTaskTextRegression(AbsTask):
    """Abstract class for regression tasks

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It
    must contain the following columns:
        text: str
        value: float
    """

    evaluator: type[LinearRegressionEvaluator] = LinearRegressionEvaluator
    regressor: SklearnRegressorModel = LinearRegression(n_jobs=-1)

    train_split: str = "train"
    label_column_name: str = "value"
    input_column_name: str = "text"
    abstask_prompt = "Predict the value of the user passage."

    n_experiments: int = 10
    n_samples: int = 2048

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.n_experiments = self.metadata_dict.get("n_experiments", self.n_experiments)
        self.n_samples = self.metadata_dict.get("n_samples", self.n_samples)

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: datasets.DatasetDict,
        hf_split: str,
        *,
        hf_subset: str | None = None,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ScoresDict:
        train_split = dataset[self.train_split]
        eval_split = dataset[hf_split]

        scores_list, test_cache = [], None
        for i in range(self.n_experiments):
            logger.info(
                "=" * 10 + f" Experiment {i + 1}/{self.n_experiments} " + "=" * 10
            )

            if self.n_samples >= len(train_split):
                train_split_sampled = train_split
            else:
                train_split_sampled = self.stratified_subsampling(
                    datasets.DatasetDict({"train": train_split}),
                    seed=self.seed + i,
                    splits=["train"],
                    label=self.label_column_name,
                    n_samples=self.n_samples,
                )["train"]

            evaluator = self.evaluator(
                train_split_sampled[self.input_column_name],
                train_split_sampled[self.label_column_name],
                eval_split[self.input_column_name],
                eval_split[self.label_column_name],
                task_name=self.metadata.name,
                hf_split=hf_split,
                hf_subset=hf_subset,
                regressor=self.regressor,
                **kwargs,
            )
            scores, test_cache = evaluator(
                model, encode_kwargs=encode_kwargs, test_cache=test_cache
            )
            scores_list.append(scores)

        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in scores_list]) for k in scores_list[0]
        }
        avg_scores["scores_per_experiment"] = scores_list
        return avg_scores

    def _add_main_score(self, scores):
        scores["main_score"] = scores[self.metadata.main_score]

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        if "random_state" in self.model.get_params():
            self.model = self.model.set_params(random_state=self.seed)

        scores = {}
        hf_subsets = self.hf_subsets
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(
                f"\nTask: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
            )

            if hf_subset not in self.dataset and hf_subset == "default":
                ds = self.dataset
            else:
                ds = self.dataset[hf_subset]
            scores[hf_subset] = self._evaluate_subset(
                model,
                ds,
                hf_split=split,
                hf_subset=hf_subset,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
            self._add_main_score(scores[hf_subset])

        return scores

    @staticmethod
    def stratified_subsampling(
        dataset_dict: datasets.DatasetDict,
        seed: int,
        splits: list[str] = ["test"],
        label: str = "value",
        n_samples: int = 2048,
        n_bins: int = 10,
    ) -> datasets.DatasetDict:
        """Subsamples the dataset with stratification by the supplied label, which is assumed to be a continuous value.
        The continuous values are bucketized into `n_bins` bins based on quantiles.
        Returns a DatasetDict object.

        Args:
            dataset_dict: the DatasetDict object.
            seed: the random seed.
            splits: the splits of the dataset.
            label: the label with which the stratified sampling is based on.
            n_samples: Optional, number of samples to subsample.
            n_bins: Optional, number of bins to bucketize the continuous label.
        """
        stratify_col_name = f"{label}_binned_for_stratification"

        for split in splits:
            if n_samples >= len(dataset_dict[split]):
                logger.debug(
                    "Subsampling not needed for split %s, as n_samples is equal or greater than the number of samples.",
                    split,
                )
                continue

            dataset = dataset_dict[split]
            labels = dataset[label]

            binned_labels = pd.qcut(labels, q=n_bins, labels=False, duplicates="drop")
            dataset_with_bins: datasets.Dataset = dataset.add_column(
                name=stratify_col_name,
                column=binned_labels.tolist(),
            )
            dataset_with_bins = dataset_with_bins.cast_column(
                stratify_col_name,
                datasets.ClassLabel(names=np.unique(binned_labels).tolist()),
            )

            subsampled_dataset = dataset_with_bins.train_test_split(
                test_size=n_samples, seed=seed, stratify_by_column=stratify_col_name
            )["test"]

            subsampled_dataset = subsampled_dataset.remove_columns([stratify_col_name])
            dataset_dict[split] = subsampled_dataset

        return dataset_dict

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RegressionDescriptiveStatistics:
        train_text = []
        if hf_subset:
            texts = self.dataset[hf_subset][split]["text"]
            values = self.dataset[hf_subset][split]["value"]
            if split != "train":
                train_text = self.dataset[hf_subset]["train"]["text"]
        elif compute_overall:
            texts = []
            values = []
            for lang_subset in self.metadata.eval_langs:
                texts.extend(self.dataset[lang_subset][split]["text"])
                values.extend(self.dataset[lang_subset][split]["value"])
                if split != "train":
                    train_text.extend(self.dataset[lang_subset]["train"]["text"])
        else:
            texts = self.dataset[split]["text"]
            values = self.dataset[split]["value"]
            if split != "train":
                train_text = self.dataset["train"]["text"]

        text_lengths = [len(t) for t in texts]
        total_text_length = sum(text_lengths)

        num_texts_in_train_val = (
            len(set(texts) & set(train_text)) if split != "train" else None
        )

        return RegressionDescriptiveStatistics(
            num_samples=len(texts),
            number_of_characters=total_text_length,
            num_texts_in_train=num_texts_in_train_val,
            min_text_length=min(text_lengths) if text_lengths else 0,
            average_text_length=(total_text_length / len(texts))
            if len(texts) > 0
            else 0,
            max_text_length=max(text_lengths) if text_lengths else 0,
            unique_text=len(set(texts)),
            min_value=min(values) if values else 0.0,
            average_value=(sum(values) / len(values)) if len(values) > 0 else 0.0,
            max_value=max(values) if values else 0.0,
        )
