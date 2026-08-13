"""Removing duplicated samples from a task."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from ._apply import filter_task_rows
from ._row_filters import keep_first_occurrence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mteb.abstasks.abstask import AbsTask
    from mteb.types import HFSubset

    from ._row_filters import TextNormalization

T = TypeVar("T", bound="AbsTask")


def remove_duplicates(
    task: T,
    *,
    normalize: TextNormalization = "strip",
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove duplicated samples from a task, keeping the first occurrence of each.

    Two samples are duplicates when all of their content columns match. Text matches when it is identical once
    surrounding whitespace is stripped, which `normalize` can loosen; images, audio and video match when their
    content hashes are equal. Duplicates are removed within each split, so a sample appearing in both the train and
    the test split is kept in both.

    The data is loaded if it has not been loaded yet, and the task's dataset is modified in place. The task is also
    returned, so filters can be chained. Because the dataset then no longer matches the published one, the task is
    marked with `data_modified` and its scores are not comparable to the results on the leaderboard.

    For a retrieval task the corpus and the queries are deduplicated together with their relevance judgements: a
    judgement pointing at a removed duplicate is moved to the copy that was kept, so no query loses a positive
    document, and any query left without one afterwards is dropped, as it cannot be scored.

    Args:
        task: The task to deduplicate.
        normalize: How much of a difference between two texts to ignore when comparing them: `"strip"` (the
            default) only ignores surrounding whitespace, `"casefold"` also ignores case, and `"alphanumeric"`
            also ignores punctuation and repeated whitespace. The looser settings catch more duplicates but can
            merge samples that a reader would tell apart, and case folding is not meaningful in every script.
        columns: The content columns to compare. Defaults to every content column of the task, e.g. `["text"]` for
            classification or `["sentence1", "sentence2"]` for pair classification.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset and for hashing non-text content.

    Returns:
        The task, so that calls can be chained.

    Raises:
        NotImplementedError: If the task does not declare which of its columns hold content.
        ValueError: If `columns`, `splits` or `subsets` select none of the task's data.

    Examples:
        >>> import mteb
        >>> from mteb.quality import remove_duplicates
        >>> task = mteb.get_task("MassiveIntentClassification")
        >>> task = remove_duplicates(task)
        >>> # also treat "Wake me up!" and "wake  me  up" as duplicates
        >>> task = remove_duplicates(task, normalize="alphanumeric")
    """
    filter_task_rows(
        task,
        keep_first_occurrence,
        filter_name="remove_duplicates",
        normalize=normalize,
        remap_duplicates=True,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
    return task
