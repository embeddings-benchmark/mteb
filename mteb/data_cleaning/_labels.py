"""Removing samples by their label: the ones that carry none, and the ones their content contradicts."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from mteb.abstasks.regression import AbsTaskRegression

from ._filtering import (
    _CleaningFilter,
    _filter_task_rows,
    _row_key,
    _strip_whitespace,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from mteb.types import HFSubset

    from ._filtering import KeepRowsFn, Normalization, T


def _keep_labelled(rows: Iterable[tuple[str, ...]]) -> list[int]:
    """Keep the rows that carry a label.

    Args:
        rows: The comparable content of each row, one tuple per row, with the row's label as its last entry.

    Returns:
        The indices of the rows whose label is not empty.
    """
    return [i for i, row in enumerate(rows) if row[-1]]


def _keep_agreeing_labels(rows: Iterable[tuple[str, ...]]) -> list[int]:
    """Keep the rows whose content carries no label other than theirs.

    Args:
        rows: The comparable content of each row, one tuple per row, with the row's label as its last entry.

    Returns:
        The indices of the rows whose content is labelled the same way wherever it occurs. A row carrying no
        label is kept, as it contradicts nothing.
    """
    labelled = [(_row_key(row[:-1]), row[-1]) for row in rows]
    labels: dict[bytes, set[str]] = defaultdict(set)
    for key, label in labelled:
        if label:
            labels[key].add(label)
    return [i for i, (key, _) in enumerate(labelled) if len(labels[key]) < 2]


def _filter_by_label(
    task: T,
    name: str,
    keep_fn: KeepRowsFn,
    *,
    normalization: Normalization,
    columns: Sequence[str] | None,
    splits: Sequence[str] | None,
    subsets: Sequence[HFSubset] | None,
    num_proc: int | None,
) -> T:
    """Apply a filter that reads the label of each sample alongside its content.

    Raises:
        NotImplementedError: If `task` scores its samples on a continuous scale, where two values differ by how
            much rather than by contradicting each other.
        ValueError: If `task` declares no label column.
    """
    if isinstance(task, AbsTaskRegression):
        raise NotImplementedError(
            f"`{name}` does not apply to '{task.metadata.name}': it scores its samples on a continuous scale, "
            "where two values are close or far apart rather than contradictory."
        )
    label_column = getattr(task, "label_column_name", None)
    if label_column is None:
        raise ValueError(
            f"`{name}` needs the labels of '{task.metadata.name}', which declares no `label_column_name`."
        )

    return _filter_task_rows(
        task,
        _CleaningFilter(name, keep_fn, label_column=label_column),
        normalization=normalization,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def remove_unlabeled(
    task: T,
    *,
    normalization: Normalization = _strip_whitespace,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove the samples of a task that carry no label, leaving nothing to score them against.

    A label is missing when it is None, a NaN, an empty text or an empty list of labels.

    Args:
        task: The task to filter. It is not modified.
        normalization: How to rewrite a text before comparing it. Labels are read as they are.
        columns: The content columns to compare. Defaults to every content column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset and for hashing non-text content.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `EmotionClassification (remove_unlabeled)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead, or scores its samples
            on a continuous scale.
        ValueError: If `task` declares no label column, or if `splits` and `subsets` together match none of its
            splits.
        KeyError: If `columns` names a column the task does not declare.

    Examples:
        >>> import mteb
        >>> from mteb.data_cleaning import remove_unlabeled
        >>> task = mteb.get_task("EmotionClassification")
        >>> cleaned = remove_unlabeled(task)
    """
    return _filter_by_label(
        task,
        "remove_unlabeled",
        _keep_labelled,
        normalization=normalization,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def remove_conflicting_labels(
    task: T,
    *,
    normalization: Normalization = _strip_whitespace,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove the samples of a task whose content occurs under more than one label, leaving no answer to give.

    Every occurrence goes, as none of them can be told to be the correct one. A list of labels is read as a set,
    so the same labels in another order agree. Splits are filtered on their own: use `remove_train_leakage` for an
    evaluation sample that the train split labels differently, and `remove_unlabeled` for the samples that carry
    no label to conflict with.

    Args:
        task: The task to filter. It is not modified.
        normalization: How to rewrite a text before comparing it. The default ignores surrounding whitespace only.
        columns: The content columns to compare. Defaults to every content column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset and for hashing non-text content.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `EmotionClassification (remove_conflicting_labels)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead, or scores its samples
            on a continuous scale, where two values differ by how much rather than by contradicting each other.
        ValueError: If `task` declares no label column, or if `splits` and `subsets` together match none of its
            splits.
        KeyError: If `columns` names a column the task does not declare.

    Examples:
        >>> import mteb
        >>> from mteb.data_cleaning import remove_conflicting_labels
        >>> task = mteb.get_task("EmotionClassification")
        >>> cleaned = remove_conflicting_labels(task)
    """
    return _filter_by_label(
        task,
        "remove_conflicting_labels",
        _keep_agreeing_labels,
        normalization=normalization,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
