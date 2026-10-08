"""Removing evaluation samples that also appear in the train split."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mteb.abstasks.retrieval import AbsTaskRetrieval

from ._filtering import (
    _CleaningFilter,
    _filter_task_rows,
    _row_key,
    _strip_whitespace,
)

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence

    from mteb.types import HFSubset

    from ._filtering import Normalization, T


def _keep_unseen(
    rows: Iterable[tuple[str, ...]], reference: Collection[bytes] = ()
) -> list[int]:
    """Keep the rows whose content does not occur in `reference`.

    Args:
        rows: The comparable content of each row, one tuple per row with one entry per compared column.
        reference: The keys of the rows to compare against, which the filter binds per subset.

    Returns:
        The indices of the rows that `reference` does not hold.
    """
    return [i for i, row in enumerate(rows) if _row_key(row) not in reference]


def remove_train_leakage(
    task: T,
    *,
    pool_subsets: bool = False,
    normalization: Normalization = _strip_whitespace,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove the samples of a task that also appear in its train split, which a model may have been trained on.

    A sample leaks when all of its content columns match a train sample's -- text as `normalization` rewrites it,
    images, audio and video by their content hash -- which is what the `samples_in_train` of the task's descriptive
    statistics reports, except that it counts the distinct leaked samples while this removes every leaking row.
    Labels are not compared, as the same content under another label leaks just as much, and each subset is compared
    against its own train split, which is itself left untouched. It is `remove_duplicates` with the train split as
    its reference, without removing the duplicates a split holds of its own.

    Args:
        task: The task to filter. It is not modified.
        pool_subsets: Whether a sample is compared against the train split of every subset, as the task's own
            `samples_in_train` counts it, rather than against its own subset's, as that subset is evaluated.
        normalization: How to rewrite a text before comparing it. The default ignores surrounding whitespace only.
        columns: The content columns to compare. Defaults to every content column of the task.
        splits: The splits to filter. Defaults to every split but the train one.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset and for hashing non-text content.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `MassiveIntentClassification (remove_train_leakage)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead, or is a retrieval task,
            whose corpus is commonly shared between splits by design rather than leaked into them.
        ValueError: If `splits` and `subsets` together match none of the task's splits, as they do for a task
            without a train split.
        KeyError: If `columns` names a column the task does not declare.

    Examples:
        >>> import mteb
        >>> from mteb.data_cleaning import remove_train_leakage
        >>> task = mteb.get_task("AmazonCounterfactualClassification")
        >>> cleaned = remove_train_leakage(task)
    """
    if isinstance(task, AbsTaskRetrieval):
        raise NotImplementedError(
            f"`remove_train_leakage` does not apply to '{task.metadata.name}': a retrieval task commonly shares "
            "its corpus between splits by design, so a document in both is not a leak."
        )

    return _filter_task_rows(
        task,
        _CleaningFilter(
            "remove_train_leakage",
            _keep_unseen,
            reference_splits=(getattr(task, "train_split", "train"),),
            pools_subsets=pool_subsets,
        ),
        normalization=normalization,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
