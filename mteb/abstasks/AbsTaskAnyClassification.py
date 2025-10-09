from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from datasets import Dataset, DatasetDict
from PIL import ImageFile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from mteb.models import Encoder, MTEBModels
from mteb.types import HFSubset, ScoresDict
from mteb.types.statistics import (
    ImageStatistics,
    LabelStatistics,
    SplitDescriptiveStatistics,
    TextStatistics,
)

from .._evaluators.classification_evaluator import (
    ClassificationEvaluator,
    SklearnClassifierProtocol,
)
from ._statistics_calculation import (
    calculate_image_statistics,
    calculate_label_statistics,
    calculate_text_statistics,
)
from .AbsTask import AbsTask

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class ClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Classification

    Attributes:
        num_samples: number of samples in the dataset.
        number_texts_intersect_with_train: Number of texts in the train split

        text_statistics: Statistics for text
        image_statistics: Statistics for images
        label_statistics: Statistics for labels
    """

    num_samples: int
    number_texts_intersect_with_train: int | None

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    label_statistics: LabelStatistics


class ClassificationMetrics(TypedDict):
    """Scores for classification tasks

    Attributes:
        accuracy: Accuracy score.
        f1: F1 score (macro).
        f1_weighted: Weighted F1 score.
        precision: Precision score (macro).
        precision_weighted: Weighted precision score.
        recall: Recall score (macro).
        recall_weighted: Weighted recall score.
        ap: Average precision score (macro) for binary classification.
        ap_weighted: Weighted average precision score for binary classification.
    """

    accuracy: float
    f1: float
    f1_weighted: float
    precision: float
    precision_weighted: float
    recall: float
    recall_weighted: float
    ap: float | None
    ap_weighted: float | None


class FullClassificationMetrics(ClassificationMetrics):
    """Full classification metrics including scores per experiment. In main scores, the average over all experiments is reported.

    Attributes:
        scores_per_experiment: List of ClassificationMetrics for each experiment.
    """

    scores_per_experiment: list[ClassificationMetrics]


class AbsTaskAnyClassification(AbsTask):
    """Abstract class for classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It
    must contain the following columns:
        input_column_name: input (str | image)
        label_column_name: int

    Attributes:
       samples_per_label: Number of samples to use pr. label. These samples are embedded and a classifier is fit using the labels and samples.

    """

    evaluator: type[ClassificationEvaluator] = ClassificationEvaluator
    classifier: SklearnClassifierProtocol = LogisticRegression(
        n_jobs=-1,
        max_iter=100,
    )

    samples_per_label: int = 8
    n_experiments: int = 10
    k: int = 3
    train_split: str = "train"
    label_column_name: str = "label"
    input_column_name: str = "text"
    abstask_prompt = "Classify user passages."

    def evaluate(
        self,
        model: MTEBModels,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        if not isinstance(model, Encoder):
            raise TypeError(
                f"Model {model} is a SearchProtocol, but this task {self.metadata.name} does not support Search. "
                "Please use a Encoder model instead."
            )

        if not self.data_loaded:
            self.load_data()

        if "random_state" in self.classifier.get_params():
            self.classifier = self.classifier.set_params(random_state=self.seed)
        scores = {}
        hf_subsets = self.hf_subsets
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            logger.info(
                f"Task: {self.metadata.name}, split: {split}, subset: {hf_subset}. Running..."
            )

            if hf_subset not in self.dataset and hf_subset == "default":
                ds = self.dataset
            else:
                ds = self.dataset[hf_subset]

            if isinstance(ds, (Dataset, DatasetDict)):
                ds = ds.select_columns([self.label_column_name, self.input_column_name])
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
    ) -> FullClassificationMetrics:
        train_split = data_split[self.train_split]
        eval_split = data_split[hf_split]
        params = {"k": self.k}
        params.update(kwargs)

        scores = []
        # we store idxs to make the shuffling reproducible
        test_cache, idxs = None, None

        all_predictions = []
        for i in range(self.n_experiments):
            logger.info(f"Running classification experiment ({i}/{self.n_experiments})")
            # Bootstrap `self.samples_per_label` samples per label for each split
            train_dataset, idxs = self._undersample_data(
                train_split,
                idxs,
            )

            evaluator = self.evaluator(
                train_dataset,
                eval_split,
                self.input_column_name,
                self.label_column_name,
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                classifier=self.classifier,
                **params,
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
            # ap will be none for non binary classification tasks
            k: (
                np.mean(values)
                if (values := [s[k] for s in scores if s[k] is not None])
                else np.nan
            )
            for k in scores[0].keys()
        }
        logger.info("Running classification - Finished.")
        return FullClassificationMetrics(
            scores_per_experiment=scores,
            **avg_scores,
        )

    def _calculate_scores(
        self,
        y_test: np.ndarray | list[int],
        y_pred: np.ndarray,
    ) -> ClassificationMetrics:
        scores = ClassificationMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred, average="macro"),
            f1_weighted=f1_score(y_test, y_pred, average="weighted"),
            precision=precision_score(y_test, y_pred, average="macro"),
            precision_weighted=precision_score(y_test, y_pred, average="weighted"),
            recall=recall_score(y_test, y_pred, average="macro"),
            recall_weighted=recall_score(y_test, y_pred, average="weighted"),
            ap=None,
            ap_weighted=None,
        )

        # if binary classification
        if len(np.unique(y_test)) == 2:
            scores["ap"] = average_precision_score(y_test, y_pred, average="macro")
            scores["ap_weighted"] = average_precision_score(
                y_test, y_pred, average="weighted"
            )
        return scores

    def _undersample_data(
        self, dataset: Dataset, idxs: list[int] | None = None
    ) -> tuple[Dataset, list[int]]:
        """Undersample data to have `samples_per_label` samples of each label.

        Args:
            dataset: Hugging Face `datasets.Dataset` containing "text" and "label".
            idxs: Optional indices to shuffle and sample from.

        Returns:
            A new Dataset containing undersampled examples.
            The shuffled indices used for sampling.
        """
        if idxs is None:
            idxs = list(range(len(dataset)))

        # using RandomState for backward compatibility with `v1`
        rng_state = np.random.RandomState(self.seed)
        rng_state.shuffle(idxs)

        label_counter: dict[str, int] = defaultdict(int)
        sampled_idxs = []

        for i in idxs:
            label = dataset[i][self.label_column_name]
            if label_counter[label] < self.samples_per_label:
                sampled_idxs.append(i)
                label_counter[label] += 1

        return dataset.select(sampled_idxs), idxs

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> ClassificationDescriptiveStatistics:
        train_text = []
        if hf_subset:
            inputs = self.dataset[hf_subset][split][self.input_column_name]
            label = self.dataset[hf_subset][split][self.label_column_name]
            if split != self.train_split:
                train_text = self.dataset[hf_subset][self.train_split][
                    self.input_column_name
                ]
        elif compute_overall:
            inputs = []
            label = []
            for hf_subset in self.metadata.eval_langs:
                inputs.extend(self.dataset[hf_subset][split][self.input_column_name])
                label.extend(self.dataset[hf_subset][split][self.label_column_name])
                if split != self.train_split:
                    train_text.extend(
                        self.dataset[hf_subset][self.train_split][
                            self.input_column_name
                        ]
                    )
        else:
            inputs = self.dataset[split][self.input_column_name]
            label = self.dataset[split][self.label_column_name]
            if split != self.train_split:
                train_text = self.dataset[self.train_split][self.input_column_name]

        image_statistics = None
        text_statistics = None
        num_texts_in_train = None

        if "image" in self.metadata.modalities:
            image_statistics = calculate_image_statistics(inputs)
        if "text" in self.metadata.modalities:
            text_statistics = calculate_text_statistics(inputs)
            num_texts_in_train = (
                len(set(inputs) & set(train_text))
                if split != self.train_split
                else None
            )

        label_statistics = calculate_label_statistics(label)

        return ClassificationDescriptiveStatistics(
            num_samples=len(inputs),
            number_texts_intersect_with_train=num_texts_in_train
            if num_texts_in_train
            else None,
            text_statistics=text_statistics,
            image_statistics=image_statistics,
            label_statistics=label_statistics,
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self._upload_dataset_to_hub(
            repo_name,
            [
                self.input_column_name,
                self.label_column_name,
            ],
        )
