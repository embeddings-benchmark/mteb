from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np
from datasets import Dataset
from sklearn import metrics

from mteb._evaluators import ZeroShotClassificationEvaluator
from mteb.models import EncoderProtocol
from mteb.types.statistics import ZeroShotClassificationDescriptiveStatistics

from ._statistics_calculation import (
    calculate_label_statistics,
    calculate_single_input_modality_statistics,
    calculate_text_statistics,
)
from .abstask import AbsTask

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from mteb.models import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import Array, EncodeKwargs, Modalities

logger = logging.getLogger(__name__)


class ZeroShotClassificationMetrics(TypedDict):
    """Metrics for ZeroShotClassification

    Attributes:
        accuracy: Accuracy of the model.
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


class AbsTaskZeroShotClassification(AbsTask):
    """Abstract class for ZeroShot Classification tasks for any modality.

    The similarity between an input (can be image or text) and candidate text prompts, such as this is a dog/this is a cat.

    Attributes:
        dataset: Huggingface dataset containing the data for the task. Dataset must contain columns specified by self.input_column_name and self.label_column_name.
        input_column_name: Name of the column containing the inputs (image or text).
        label_column_name: Name of the column containing the labels. Labels must be
            integer indices of the candidate labels or strings matching an entry of
            `get_candidate_labels`.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    input_column_name: str | Sequence[Modalities] = "image"
    label_column_name: str = "label"
    abstask_prompt = "Classify the given input into one of the candidate labels."

    def dataset_transform(self, num_proc: int | None = None, **kwargs: Any) -> None:
        """Keep only eval splits. Zero-shot tasks don't need train splits."""
        if self.dataset is None:
            return
        splits_to_keep = set(self.metadata.eval_splits)
        for split in list(self.dataset.keys()):
            if split not in splits_to_keep:
                del self.dataset[split]

    def _calculate_descriptive_statistics_from_split(
        self,
        split: str,
        *,
        hf_subset: str | None = None,
        compute_overall: bool = False,
        num_proc: int | None = None,
    ) -> ZeroShotClassificationDescriptiveStatistics:
        if isinstance(self.input_column_name, str):
            col_map = {self.metadata.modalities[0]: self.input_column_name}
        else:
            col_map = {col: col for col in self.input_column_name}

        if hf_subset:
            ds = self.dataset[hf_subset][split]
            col_inputs = {mod: ds[col] for mod, col in col_map.items()}
            labels = ds[self.label_column_name]
        elif compute_overall:
            col_inputs = {mod: [] for mod in col_map}
            labels = []
            for subset in self.metadata.eval_langs:
                ds = self.dataset[subset][split]
                for mod, col in col_map.items():
                    col_inputs[mod].extend(ds[col])
                labels.extend(ds[self.label_column_name])
        else:
            ds = self.dataset[split]
            col_inputs = {mod: ds[col] for mod, col in col_map.items()}
            labels = ds[self.label_column_name]

        modality_stats = calculate_single_input_modality_statistics(
            col_inputs, max_workers=num_proc
        )
        return ZeroShotClassificationDescriptiveStatistics(
            num_samples=len(ds[self.label_column_name]),
            **modality_stats,
            label_statistics=calculate_label_statistics(labels),
            candidates_labels_text_statistics=calculate_text_statistics(
                self.get_candidate_labels()
            ),
        )

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack,
        **kwargs: Any,
    ) -> ZeroShotClassificationMetrics:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        candidate_labels = self.get_candidate_labels()
        data_split = data_split.select_columns(
            [self.input_column_name, self.label_column_name]
            if isinstance(self.input_column_name, str)
            else [*self.input_column_name, self.label_column_name]
        )
        evaluator = ZeroShotClassificationEvaluator(
            data_split,
            self.input_column_name,
            candidate_labels,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            timer=timer,
            **kwargs,
        )
        probs = evaluator(
            model,
            encode_kwargs=encode_kwargs,
            num_proc=num_proc,
        )

        if prediction_folder:
            self._save_task_predictions(
                probs.tolist(),
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        return self._calculate_scores(
            self._normalize_labels(
                data_split[self.label_column_name],
                candidate_labels,
            ),
            probs,
        )

    @staticmethod
    def _normalize_labels(
        labels: list[int] | list[str], candidate_labels: list[str]
    ) -> list[int]:
        """Convert dataset labels to integer indices of the candidate labels.

        Predictions are always integer indices into ``candidate_labels``, while
        datasets store labels either as integer class indices (e.g. a
        ``ClassLabel`` column) or as strings. scikit-learn >= 1.9 raises an
        error when ``y_true`` contains strings and ``y_pred`` is numeric, so
        string labels are mapped to their index in ``candidate_labels``.

        Args:
            labels: Labels as stored in the dataset.
            candidate_labels: Candidate labels returned by `get_candidate_labels`.

        Returns:
            Labels as integer indices into ``candidate_labels``.

        Raises:
            ValueError: If a string label does not match any candidate label.
        """
        if not labels or not isinstance(labels[0], str):
            return cast("list[int]", labels)

        label_to_index = {label: idx for idx, label in enumerate(candidate_labels)}
        unknown_labels = sorted(set(labels) - label_to_index.keys())
        if unknown_labels:
            raise ValueError(
                "String labels must match an entry of `get_candidate_labels` to "
                f"be mapped to a candidate index, but {unknown_labels} do not. "
                "Alternatively, store labels as integer indices of the candidate "
                "labels."
            )
        return [label_to_index[label] for label in labels]

    def _calculate_scores(  # noqa: PLR6301
        self,
        labels: list[int],
        probs: Array,
    ) -> ZeroShotClassificationMetrics:
        predictions = probs.argmax(1)
        scores = ZeroShotClassificationMetrics(
            accuracy=metrics.accuracy_score(labels, predictions),
            f1=metrics.f1_score(labels, predictions, average="macro"),
            f1_weighted=metrics.f1_score(labels, predictions, average="weighted"),
            precision=metrics.precision_score(labels, predictions, average="macro"),
            precision_weighted=metrics.precision_score(
                labels, predictions, average="weighted"
            ),
            recall=metrics.recall_score(labels, predictions, average="macro"),
            recall_weighted=metrics.recall_score(
                labels, predictions, average="weighted"
            ),
            ap=None,
            ap_weighted=None,
        )

        if len(np.unique(labels)) == 2:
            positive_scores = probs[:, 1]
            scores["ap"] = metrics.average_precision_score(
                labels, positive_scores, average="macro"
            )
            scores["ap_weighted"] = metrics.average_precision_score(
                labels, positive_scores, average="weighted"
            )
        return scores

    def _push_dataset_to_hub(
        self,
        repo_name: str,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._upload_dataset_to_hub(
            repo_name,
            [
                self.input_column_name,
                self.label_column_name,
            ]
            if isinstance(self.input_column_name, str)
            else [
                *self.input_column_name,
                self.label_column_name,
            ],
            num_proc=num_proc,
        )
        labels_dataset = Dataset.from_dict({"labels": self.get_candidate_labels()})
        labels_dataset.push_to_hub(repo_name, config_name="labels", **kwargs)

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        raise NotImplementedError("This method should be overridden by subclasses")
