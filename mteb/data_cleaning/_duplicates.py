"""Removing samples that repeat the content of an earlier one."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def _keep_first_occurrence(rows: Iterable[tuple[str, ...]]) -> list[int]:
    """Keep the rows whose content has not been seen before.

    Args:
        rows: The comparable content of each row, one tuple per row with one entry per compared column.

    Returns:
        The indices of the first occurrence of each distinct row.
    """
    seen: set[bytes] = set()
    keep = []
    for i, row in enumerate(rows):
        key = _row_key(row)
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


def remove_duplicates(
    task: T,
    *,
    normalization: Normalization = _strip_whitespace,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove duplicated samples from a task, keeping the first occurrence of each.

    Two samples are duplicates when all of their content columns match. Text matches when `normalization` rewrites
    both to the same string; images, audio and video match when their content hashes are equal. Duplicates are
    removed within each split, so a sample appearing in both the train and the test split is kept in both.

    The task passed in is left untouched, and a cleaned copy is returned. The copy is named after the filters
    applied to it, e.g. `MassiveIntentClassification (remove_duplicates)`, so that its scores are recorded against
    that id rather than against the published dataset.

    For a retrieval task the corpus and the queries are deduplicated together with their relevance judgements: a
    judgement pointing at a removed duplicate is moved to the copy that was kept, so no query loses a positive
    document, and any query left without one afterwards is dropped, as it cannot be scored. Both are compared as
    the model reads them, so a document's title is part of its text, and two queries differing only in their
    instruction are not duplicates.

    Args:
        task: The task to deduplicate. It is not modified.
        normalization: How to rewrite a text before comparing it. The default ignores surrounding whitespace only.
            Looser comparisons catch more duplicates but can merge samples that a reader would tell apart, so
            prefer the narrowest one that finds the duplicates you care about.
        columns: The content columns to compare. Defaults to every content column of the task, e.g. `["text"]` for
            classification or `["sentence1", "sentence2"]` for pair classification.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset and for hashing non-text content.

    Returns:
        A copy of the task holding the deduplicated data.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If `splits` and `subsets` together match none of the task's splits, which would otherwise
            filter nothing at all.
        KeyError: If `columns` names a column the task does not declare.

    Examples:
        >>> import mteb
        >>> from mteb.data_cleaning import remove_duplicates
        >>> task = mteb.get_task("MassiveIntentClassification")
        >>> cleaned = remove_duplicates(task)
        >>> # ignore case too, so that "Wake me up!" and "wake me up!" are duplicates
        >>> cleaned = remove_duplicates(task, normalization=lambda t: t.strip().casefold())
    """
    return _filter_task_rows(
        task,
        _CleaningFilter(
            "remove_duplicates", _keep_first_occurrence, removes_duplicates=True
        ),
        normalization=normalization,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
