import logging
from pathlib import Path
from typing import Any, TypedDict

import datasets
import numpy as np
import pandas as pd
from datasets import DatasetDict
from scipy.stats import kendalltau
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mteb._evaluators.sklearn_evaluator import SklearnEvaluator, SklearnModelProtocol
from mteb.abstasks._statistics_calculation import (
    calculate_score_statistics,
    calculate_text_statistics,
)
from mteb.models.models_protocols import Encoder
from mteb.types.statistics import (
    ScoreStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

from .AbsTaskAnyClassification import AbsTaskAnyClassification

logger = logging.getLogger(__name__)


class RegressionDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Regression

    Attributes:
      num_samples: number of samples in the dataset.
      number_of_characters: Total number of symbols in the dataset.
      num_texts_in_train: Number of texts in the train split

      text_statistics: Statistics of texts

      values_statistics: Statistics of values
    """

    num_samples: int
    number_of_characters: int
    num_texts_in_train: int | None

    text_statistics: TextStatistics
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


class FullRegressionMetrics(RegressionMetrics):
    """Full Regression metrics including scores per experiment. In main scores, the average over all experiments is reported.

    Attributes:
        scores_per_experiment: List of ClassificationMetrics for each experiment.
    """

    scores_per_experiment: list[RegressionMetrics]


class AbsTaskTextRegression(AbsTaskAnyClassification):
    """Abstract class for regression tasks

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It
    must contain the following columns:
        text: str
        value: float
    """

    evaluator: type[SklearnModelProtocol] = SklearnEvaluator
    evaluator_model: SklearnModelProtocol = LinearRegression(n_jobs=-1)

    train_split: str = "train"
    label_column_name: str = "value"
    input_column_name: str = "text"
    abstask_prompt = "Predict the value of the user passage."

    n_experiments: int = 10
    n_samples: int = 2048

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: DatasetDict,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> FullRegressionMetrics:
        train_split = data_split[self.train_split]
        eval_split = data_split[hf_split]

        scores_list, test_cache = [], None
        all_predictions = []
        scores = []
        for i in range(self.n_experiments):
            logger.info(f"Running regression experiment ({i}/{self.n_experiments})")

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
                train_split_sampled,
                eval_split,
                self.input_column_name,
                self.label_column_name,
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                evaluator_model=self.evaluator_model,
                **kwargs,
            )
            y_pred, test_cache = evaluator(
                model, encode_kwargs=encode_kwargs, test_cache=test_cache
            )
            if prediction_folder:
                all_predictions.append(y_pred.tolist())
            y_test = eval_split[self.label_column_name]
            scores_exp = self._calculate_scores(y_test, y_pred)
            scores.append(scores_exp)

        if prediction_folder:
            self._save_task_predictions(
                all_predictions,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        avg_scores: dict[str, Any] = {
            k: float(np.mean([s[k] for s in scores])) for k in scores[0].keys()
        }
        return FullRegressionMetrics(
            scores_per_experiment=scores_list,
            **avg_scores,
        )

    def _calculate_scores(
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

    def _calculate_descriptive_statistics_from_split(
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

        text_lengths = [len(t) for t in texts]
        total_text_length = sum(text_lengths)

        num_texts_in_train_val = (
            len(set(texts) & set(train_text)) if split != self.train_split else None
        )

        return RegressionDescriptiveStatistics(
            num_samples=len(texts),
            number_of_characters=total_text_length,
            num_texts_in_train=num_texts_in_train_val,
            text_statistics=calculate_text_statistics(texts),
            values_statistics=calculate_score_statistics(values),
        )
