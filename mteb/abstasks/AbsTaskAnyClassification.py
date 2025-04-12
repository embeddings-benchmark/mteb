from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict

from mteb.abstasks.TaskMetadata import DescriptiveStatistics
from mteb.encoder_interface import Encoder

from ..evaluation.evaluators import (
    logRegClassificationEvaluator,
)
from ..evaluation.evaluators.ClassificationEvaluator import AbsClassificationEvaluator
from ..load_results.task_results import HFSubset, ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class ClassificationDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Classification

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        number_texts_intersect_with_train: Number of texts in the train split

        min_text_length: Minimum length of text
        average_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_texts: Number of unique texts

        min_labels_per_text: Minimum number of labels per text
        average_label_per_text: Average number of labels per text
        max_labels_per_text: Maximum number of labels per text

        min_image_width: Minimum width of images
        average_image_width: Average width of images
        max_image_width: Maximum width of images

        min_image_height: Minimum height of images
        average_image_height: Average height of images
        max_image_height: Maximum height of images


        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    num_samples: int
    number_of_characters: int
    number_texts_intersect_with_train: int | None

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_texts: int

    min_labels_per_text: int
    average_label_per_text: float
    max_labels_per_text: int

    min_image_width: float | None
    average_image_width: float | None
    max_image_width: float | None

    min_image_height: float | None
    average_image_height: float | None
    max_image_height: float | None

    unique_labels: int
    labels: dict[str, dict[str, int]]


class AbsTaskAnyClassification(AbsTask):
    """Abstract class for classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It
    must contain the following columns:
        text: str
        label: int

    Attributes:
       samples_per_label: Number of samples to use pr. label. These samples are embedded and a classifier is fit using the labels and samples.

    """

    evaluator: type[AbsClassificationEvaluator] = logRegClassificationEvaluator
    samples_per_label: int = 8
    n_experiments: int = 10
    k: int = 3
    train_split: str = "train"
    label_column_name: str = "label"
    values_column_name: str = "text"
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

        is_image = False
        if "image" in self.metadata.modalities:
            is_image = True

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
                self.values_column_name,
                self.label_column_name,
                is_image,
                task_metadata=self.metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
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

        rng_state = np.random.default_rng(self.seed)
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
            values = self.dataset[hf_subset][split][self.values_column_name]
            label = self.dataset[hf_subset][split][self.label_column_name]
            if split != self.train_split:
                train_text = self.dataset[hf_subset][self.train_split][
                    self.values_column_name
                ]
        elif compute_overall:
            values = []
            label = []
            for hf_subset in self.metadata.eval_langs:
                values.extend(self.dataset[hf_subset][split][self.values_column_name])
                label.extend(self.dataset[hf_subset][split][self.label_column_name])
                if split != self.train_split:
                    train_text.extend(
                        self.dataset[hf_subset][self.train_split][
                            self.values_column_name
                        ]
                    )
        else:
            values = self.dataset[split][self.values_column_name]
            label = self.dataset[split][self.label_column_name]
            if split != self.train_split:
                train_text = self.dataset[self.train_split][self.values_column_name]

        total_text_len = 0
        text_len = None
        img_widths, img_heights = None, None
        num_texts_in_train = None

        if "image" in self.metadata.modalities:
            img_widths, img_heights = [], []
            for img in values:
                width, height = img.size  # type: ignore
                img_heights.append(height)
                img_widths.append(width)
        else:
            text_len = [len(t) for t in values]
            total_text_len = sum(text_len)
            num_texts_in_train = (
                len(set(values) & set(train_text))
                if split != self.train_split
                else None
            )

        if isinstance(label[0], int):
            label_len = [1] * len(label)
            total_label_len = len(label)
            total_labels = label
        else:
            # multilabel classification
            label_len = [len(l) for l in label]
            total_label_len = sum(label_len)
            total_labels = []
            for l in label:
                total_labels.extend(l if len(l) > 0 else [None])

        label_count = Counter(total_labels)

        return ClassificationDescriptiveStatistics(
            num_samples=len(values),
            # text
            number_of_characters=total_text_len,
            number_texts_intersect_with_train=num_texts_in_train
            if num_texts_in_train
            else None,
            min_text_length=min(text_len) if text_len else None,
            average_text_length=total_text_len / len(values) if text_len else None,
            max_text_length=max(text_len) if text_len else None,
            unique_texts=len(set(values)) if text_len else None,
            # image
            min_image_width=min(img_widths) if img_widths else None,
            average_image_width=sum(img_widths) / len(img_widths)
            if img_widths
            else None,
            max_image_width=max(img_widths) if img_widths else None,
            min_image_height=min(img_heights) if img_heights else None,
            average_image_height=sum(img_heights) / len(img_heights)
            if img_heights
            else None,
            max_image_height=max(img_heights) if img_heights else None,
            # labels
            min_labels_per_text=min(label_len),
            average_label_per_text=total_label_len / len(label),
            max_labels_per_text=max(label_len),
            unique_labels=len(label_count),
            labels={
                str(label): {
                    "count": value,
                }
                for label, value in label_count.items()
            },
        )

    def _push_dataset_to_hub(self, repo_name: str) -> None:
        self._upload_dataset_to_hub(
            repo_name,
            [
                self.values_column_name,
                self.label_column_name,
            ],
        )
