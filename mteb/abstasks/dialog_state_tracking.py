from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset, DatasetDict
from tqdm import tqdm

from mteb.models.models_protocols import EncoderProtocol
from mteb.types import ScoresDict

from .classification import AbsTaskClassification

logger = logging.getLogger(__name__)


class AbsTaskDST(AbsTaskClassification):
    """Abstract class for Dialog State Tracking tasks."""

    classification_columns: Sequence[str]

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: DatasetDict,
        *,
        encode_kwargs: dict[str, Any],
        hf_split: str,
        hf_subset: str,
        prediction_folder: Path | None = None,
        **kwargs: Any,
    ) -> ScoresDict:
        train_split = data_split[self.train_split]
        eval_split = data_split[hf_split]
        total_scores = {}

        for column in tqdm(self.classification_columns):
            current_train_split = train_split.rename_column(column, "label")
            current_eval_split = eval_split.rename_column(column, "label")

            scores = []
            test_cache, idxs = (
                None,
                None,
            )  # we store idxs to make the shuffling reproducible

            for i in range(self.n_experiments):
                logger.info(f"Running experiment ({i}/{self.n_experiments})")
                # Bootstrap `self.samples_per_label` samples per label for each split
                train_dataset, idxs = self._undersample_data(
                    current_train_split,
                    self.samples_per_label,
                    idxs,
                )
                evaluator = self.evaluator(
                    train_dataset,
                    current_eval_split,
                    self.input_column_name,
                    "label",
                    task_metadata=self.metadata,
                    hf_split=hf_split,
                    hf_subset=hf_subset,
                    evaluator_model=self.evaluator_model,
                )

                y_pred, test_cache = evaluator(
                    model, encode_kwargs=encode_kwargs, test_cache=test_cache
                )
                # if prediction_folder:
                #     all_predictions.append(y_pred.tolist())
                y_test = current_eval_split["label"]
                scores_exp = self._calculate_scores(y_test, y_pred)
                scores.append(scores_exp)

            avg_scores: dict[str, Any] = {
                # ap will be none for non binary classification tasks
                k: (
                    float(np.mean(values))
                    if (values := [s[k] for s in scores if s[k] is not None])
                    else np.nan
                )
                for k in scores[0].keys()
            }
            avg_scores["scores_per_experiment"] = scores
            total_scores[column] = avg_scores
        for metric in ["f1", "accuracy"]:
            total_scores[metric] = np.mean(
                [total_scores[column][metric] for column in self.classification_columns]
            )
        return total_scores

    def _undersample_data(
        self, dataset: Dataset, samples_per_label: int, idxs=None
    ) -> tuple[Dataset, list[int]]:
        """Undersample data to have `samples_per_label` samples of each label.

        Args:
            dataset: Hugging Face `datasets.Dataset` containing "text" and "label".
            samples_per_label: Number of samples per label to retain.
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
            label = dataset[i]["label"]
            if label_counter[label] < samples_per_label:
                sampled_idxs.append(i)
                label_counter[label] += 1

        return dataset.select(sampled_idxs), idxs
