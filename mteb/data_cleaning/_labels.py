"""Removing samples whose label says nothing: a missing one, or one that the same content contradicts."""

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

    from ._filtering import Normalization, T


def _keep_clear_labels(rows: Iterable[tuple[str, ...]]) -> list[int]:
    """Keep the rows that carry a label, and whose content carries no other one.

    Args:
        rows: The comparable content of each row, one tuple per row, with the row's label as its last entry.

    Returns:
        The indices of the labelled rows whose content is labelled the same way wherever it occurs.
    """
    labelled = [(_row_key(row[:-1]), row[-1]) for row in rows]
    labels: dict[bytes, set[str]] = defaultdict(set)
    for key, label in labelled:
        if label:
            labels[key].add(label)
    return [
        i for i, (key, label) in enumerate(labelled) if label and len(labels[key]) == 1
    ]


def remove_unclear_labels(
    task: T,
    *,
    normalization: Normalization = _strip_whitespace,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove the samples of a task that carry no label, or whose content carries a contradicting one.

    A label is missing when it is None, an empty text or an empty list of labels, leaving nothing to score the
    sample against. It contradicts when the same content occurs under another label within the split, which leaves
    no answer a model could get right; every occurrence goes, as none of them can be told to be the correct one. A
    list of labels is read as a set, so the same labels in another order agree. Splits are filtered on their own:
    use `remove_train_leakage` for an evaluation sample that the train split labels differently.

    Args:
        task: The task to filter. It is not modified.
        normalization: How to rewrite a text before comparing it. The default ignores surrounding whitespace only.
        columns: The content columns to compare. Defaults to every content column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset and for hashing non-text content.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `EmotionClassification (remove_unclear_labels)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead, or scores its samples
            on a continuous scale, where two values differ by how much rather than by contradicting each other.
        ValueError: If `task` declares no label column, or if `splits` and `subsets` together match none of its
            splits.
        KeyError: If `columns` names a column the task does not declare.

    Examples:
        >>> import mteb
        >>> from mteb.data_cleaning import remove_unclear_labels
        >>> task = mteb.get_task("EmotionClassification")
        >>> cleaned = remove_unclear_labels(task)
    """
    if isinstance(task, AbsTaskRegression):
        raise NotImplementedError(
            f"`remove_unclear_labels` does not apply to '{task.metadata.name}': it scores its samples on a "
            "continuous scale, where two values are close or far apart rather than contradictory."
        )
    label_column = getattr(task, "label_column_name", None)
    if label_column is None:
        raise ValueError(
            f"`remove_unclear_labels` needs the labels of '{task.metadata.name}', which declares no "
            "`label_column_name`."
        )

    return _filter_task_rows(
        task,
        _CleaningFilter(
            "remove_unclear_labels",
            _keep_clear_labels,
            label_column=label_column,
        ),
        normalization=normalization,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
