from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from PIL import ImageFile
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from mteb.models.models_protocols import Encoder
from mteb.types import HFSubset, ScoresDict
from mteb.types.statistics import (
    DescriptiveStatistics,
    ImageStatistics,
    LabelStatistics,
    TextStatistics,
)

from ..evaluation.evaluators.classification_evaluator import ClassificationEvaluator
from ._statistics_calculation import (
    calculate_image_statistics,
    calculate_label_statistics,
    calculate_text_statistics,
)
from .AbsTask import AbsTask

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger(__name__)


class ClassificationDescriptiveStatistics(DescriptiveStatistics):
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
    classifier: BaseEstimator = LogisticRegression(
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
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
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
        dataset: DatasetDict,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: dict[str, Any],
        **kwargs,
    ) -> ScoresDict:
        train_split = dataset[self.train_split]
        eval_split = dataset[hf_split]
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
            scores_exp, test_cache = evaluator(
                model, encode_kwargs=encode_kwargs, test_cache=test_cache
            )
            scores.append(scores_exp)

        avg_scores: dict[str, Any] = {
            k: np.mean([s[k] for s in scores]) for k in scores[0].keys()
        }
        avg_scores["scores_per_experiment"] = scores
        return avg_scores

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

        label_counter = defaultdict(int)
        sampled_idxs = []

        for i in idxs:
            label = dataset[i][self.label_column_name]
            if label_counter[label] < self.samples_per_label:
                sampled_idxs.append(i)
                label_counter[label] += 1

        return dataset.select(sampled_idxs), idxs

    def _calculate_metrics_from_split(
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
