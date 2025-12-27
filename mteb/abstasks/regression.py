import logging
from typing import TypedDict

import datasets
import numpy as np
import pandas as pd
from datasets import Dataset
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mteb._evaluators.sklearn_evaluator import SklearnEvaluator, SklearnModelProtocol
from mteb.abstasks._statistics_calculation import (
    calculate_image_statistics,
    calculate_score_statistics,
    calculate_text_statistics,
)
from mteb.types.statistics import (
    ImageStatistics,
    ScoreStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

from .classification import AbsTaskClassification

logger = logging.getLogger(__name__)


class RegressionDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Regression

    Attributes:
      num_samples: number of samples in the dataset.
      num_texts_in_train: Number of texts in the train split

      text_statistics: Statistics of texts
      image_statistics: Statistics of images

      values_statistics: Statistics of values
    """

    num_samples: int
    num_texts_in_train: int | None

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    values_statistics: ScoreStatistics


class RegressionMetrics(TypedDict):
    """Regression metrics.

    Attributes:
        mae: Mean Absolute Error.
        mse: Mean Squared Error.
        rmse: Root Mean Squared Error.
        r2: R^2 (coefficient of determination) regression score function.
        kendalltau: Kendall's tau correlation coefficient.
    """

    mae: float
    mse: float
    rmse: float
    r2: float
    kendalltau: float


class AbsTaskRegression(AbsTaskClassification):
    """Abstract class for regression tasks

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It
    must contain the following columns:
        text: str
        value: float

    Attributes:
        dataset: A HuggingFace Dataset containing the data for the regression task. It must contain the following columns: input_column_name and label_column_name.
            Input can be any text or images, and label must be a continuous value.
        input_column_name: Name of the column containing the text inputs.
        label_column_name: Name of the column containing the continuous values.
        train_split: Name of the training split in the dataset.
        n_experiments: Number of experiments to run with different random seeds.
        n_samples: Number of samples to use for training the regression model. If the dataset has fewer samples than n_samples, all samples are used.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
        evaluator_model: The model to use for evaluation. Can be any sklearn compatible model. Default is `LinearRegression`.
            Full details of api in [`SklearnModelProtocol`][mteb._evaluators.sklearn_evaluator.SklearnModelProtocol].
    """

    evaluator: type[SklearnEvaluator] = SklearnEvaluator
    evaluator_model: SklearnModelProtocol = LinearRegression(n_jobs=-1)

    train_split: str = "train"
    label_column_name: str = "value"
    input_column_name: str = "text"
    abstask_prompt = "Predict the value of the user passage."

    n_experiments: int = 10
    n_samples: int = 2048

    def _undersample_data(
        self, dataset: Dataset, experiment_num: int, idxs: list[int] | None = None
    ) -> tuple[Dataset, list[int]]:
        if self.n_samples >= len(dataset):
            train_split_sampled = dataset
        else:
            train_split_sampled = self.stratified_subsampling(
                datasets.DatasetDict({"train": dataset}),
                seed=self.seed + experiment_num,
                splits=["train"],
                label=self.label_column_name,
                n_samples=self.n_samples,
            )["train"]
        return train_split_sampled, []

    def _calculate_scores(  # type: ignore[override]
        self,
        y_test: np.ndarray | list[int],
        y_pred: np.ndarray,
    ) -> RegressionMetrics:
        mse = mean_squared_error(y_test, y_pred)
        return RegressionMetrics(
            mse=mse,
            mae=mean_absolute_error(y_test, y_pred),
            r2=r2_score(y_test, y_pred),
            kendalltau=kendalltau(y_test, y_pred).statistic,
            rmse=np.sqrt(mse),
        )

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

        Args:
            dataset_dict: the DatasetDict object.
            seed: the random seed.
            splits: the splits of the dataset.
            label: the label with which the stratified sampling is based on.
            n_samples: Optional, number of samples to subsample.
            n_bins: Optional, number of bins to bucketize the continuous label.

        Returns:
            A subsampled DatasetDict object.
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

    def _calculate_descriptive_statistics_from_split(  # type: ignore[override]
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RegressionDescriptiveStatistics:
        train_text = []
        if hf_subset:
            texts = self.dataset[hf_subset][split][self.input_column_name]
            values = self.dataset[hf_subset][split][self.label_column_name]
            if split != self.train_split:
                train_text = self.dataset[hf_subset][self.train_split][
                    self.input_column_name
                ]
        elif compute_overall:
            texts = []
            values = []
            for lang_subset in self.metadata.eval_langs:
                texts.extend(self.dataset[lang_subset][split][self.input_column_name])
                values.extend(self.dataset[lang_subset][split][self.label_column_name])
                if split != "train":
                    train_text.extend(
                        self.dataset[lang_subset][self.train_split][
                            self.input_column_name
                        ]
                    )
        else:
            texts = self.dataset[split][self.input_column_name]
            values = self.dataset[split][self.label_column_name]
            if split != "train":
                train_text = self.dataset[self.train_split][self.input_column_name]

        text_statistics = None
        image_statistics = None
        num_texts_in_train = None
        if self.metadata.modalities == ["text"]:
            text_statistics = calculate_text_statistics(texts)
            num_texts_in_train = (
                len(set(texts) & set(train_text)) if split != self.train_split else None
            )
        elif self.metadata.modalities == ["image"]:
            image_statistics = calculate_image_statistics(texts)

        return RegressionDescriptiveStatistics(
            num_samples=len(texts),
            num_texts_in_train=num_texts_in_train,
            text_statistics=text_statistics,
            image_statistics=image_statistics,
            values_statistics=calculate_score_statistics(values),
        )
