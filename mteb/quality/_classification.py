"""Checks that only apply to classification tasks, whose labels a filter can leave unusable."""

from __future__ import annotations

import logging
from collections import Counter
from typing import TYPE_CHECKING

from mteb.abstasks.multilabel_classification import AbsTaskMultilabelClassification
from mteb.abstasks.regression import AbsTaskRegression

from ._filters import _datasets_by_subset, _warn

if TYPE_CHECKING:
    from mteb.abstasks.classification import AbsTaskClassification

logger = logging.getLogger(__name__)


def _warn_about_label_distribution(
    task: AbsTaskClassification, *, min_examples_per_label: int
) -> None:
    """Report labels that filtering left too rare to train on, or absent from the train split.

    Args:
        task: The classification task to check.
        min_examples_per_label: How many examples of a label the train split needs for the task to be trainable.
    """
    if isinstance(task, AbsTaskMultilabelClassification | AbsTaskRegression):
        # the labels are a list per row or a continuous value, so counting how often each occurs says nothing
        return

    for subset, dataset_dict in _datasets_by_subset(task).items():
        if task.train_split not in dataset_dict:
            continue
        train_labels = Counter(dataset_dict[task.train_split][task.label_column_name])

        too_rare = sorted(
            str(label)
            for label, count in train_labels.items()
            if count < min_examples_per_label
        )
        if too_rare:
            _warn(
                f"The '{task.train_split}' split of '{task.metadata.name}' (subset '{subset}') has fewer than "
                f"{min_examples_per_label} examples for the labels {too_rare}, which stratified sampling cannot "
                "handle."
            )

        for split, dataset in dataset_dict.items():
            if split == task.train_split:
                continue
            unseen = sorted(
                str(label)
                for label in set(dataset[task.label_column_name]) - set(train_labels)
            )
            if unseen:
                _warn(
                    f"The '{split}' split of '{task.metadata.name}' (subset '{subset}') contains the labels "
                    f"{unseen}, which no longer occur in '{task.train_split}' and can never be predicted."
                )
