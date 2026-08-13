"""Applying a row filter to a whole task, across its subsets and splits."""

from __future__ import annotations

import logging
import warnings
from collections import Counter
from collections.abc import Mapping  # noqa: TC003
from typing import TYPE_CHECKING, Any, cast

from datasets import Dataset

from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.multilabel_classification import AbsTaskMultilabelClassification
from mteb.abstasks.regression import AbsTaskRegression
from mteb.abstasks.retrieval import AbsTaskRetrieval

from ._retrieval import filter_retrieval_split
from ._row_filters import apply_row_filter

if TYPE_CHECKING:
    from collections.abc import Sequence

    from datasets import DatasetDict

    from mteb.abstasks.abstask import AbsTask
    from mteb.types import HFSubset, Modalities

    from ._row_filters import KeepIndicesFn, TextNormalization

logger = logging.getLogger(__name__)


def _warn(msg: str) -> None:
    logger.warning(msg)
    warnings.warn(msg, stacklevel=3)


def mark_data_modified(task: AbsTask) -> None:
    """Record that the task's data no longer matches the published dataset, warning the first time."""
    if task.data_modified:
        return
    task.data_modified = True

    msg = (
        f"The data of '{task.metadata.name}' was modified locally, so it no longer matches revision "
        f"{task.metadata.revision} of the published dataset. Scores computed from it are not comparable to other "
        "results and must not be submitted to the leaderboard."
    )
    if task.metadata.descriptive_stats is not None:
        msg += " Its descriptive statistics still describe the published dataset."
    _warn(msg)


def datasets_by_subset(task: AbsTask) -> dict[HFSubset, DatasetDict]:
    """`task.dataset` normalized to a `{subset: {split: Dataset}}` mapping.

    Monolingual tasks store their data as a plain `{split: Dataset}` mapping; that mapping is returned under the
    `"default"` subset. The returned `DatasetDict`s are the task's own, so assigning to them updates the task.
    """
    if task.dataset is None:
        raise ValueError(f"Dataset of task '{task.metadata.name}' is not loaded.")

    first_value = next(iter(task.dataset.values()), None)
    if isinstance(first_value, Dataset):
        return {"default": cast("DatasetDict", task.dataset)}
    return task.dataset


def _no_split_matched_message(
    task_name: str, available: Mapping[str, Mapping[str, Any]]
) -> str:
    """The message raised when the `splits`/`subsets` given to a filter select nothing."""
    listed = ", ".join(
        f"{subset}: {sorted(splits)}" for subset, splits in available.items()
    )
    return f"The given splits and subsets do not select any data of '{task_name}'. The task has {listed}."


def _warn_about_label_distribution(task: AbsTaskClassification) -> None:
    """Warn about labels that filtering left too rare to train on, or absent from the train split."""
    if isinstance(task, AbsTaskMultilabelClassification | AbsTaskRegression):
        # the labels are a list per row or a continuous value, so counting how often each occurs says nothing
        return

    for subset, dataset_dict in datasets_by_subset(task).items():
        if task.train_split not in dataset_dict:
            continue
        train_labels = Counter(dataset_dict[task.train_split][task.label_column_name])

        # stratified subsampling and the train/test split both need at least two examples per label
        too_rare = sorted(
            str(label) for label, count in train_labels.items() if count < 2
        )
        if too_rare:
            _warn(
                f"The '{task.train_split}' split of '{task.metadata.name}' (subset '{subset}') has fewer than two "
                f"examples for the labels {too_rare}, which stratified sampling cannot handle."
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


def warn_about_unusable_data(task: AbsTask) -> None:
    """Warn about data that a filter left in a state the evaluators cannot handle."""
    if isinstance(task, AbsTaskRetrieval):
        for subset, splits_data in task.dataset.items():
            for split, split_data in splits_data.items():
                empty = sorted(
                    name
                    for name in ("corpus", "queries")
                    if len(split_data[name]) == 0  # type: ignore[literal-required]
                )
                if empty:
                    _warn(
                        f"Filtering left the {' and the '.join(empty)} of the '{split}' split of "
                        f"'{task.metadata.name}' (subset '{subset}') empty. Evaluating it will fail."
                    )
        return

    for subset, dataset_dict in datasets_by_subset(task).items():
        empty = sorted(split for split, ds in dataset_dict.items() if len(ds) == 0)
        if empty:
            _warn(
                f"Filtering left the splits {empty} of '{task.metadata.name}' (subset '{subset}') empty. "
                "Evaluating them will fail."
            )

    if isinstance(task, AbsTaskClassification):
        _warn_about_label_distribution(task)


def _resolve_columns(
    task: AbsTask, filter_name: str, columns: Sequence[str] | None
) -> dict[str, Modalities]:
    col_modalities = task._get_content_columns()
    if columns is not None:
        unknown = [column for column in columns if column not in col_modalities]
        if unknown:
            raise ValueError(
                f"'{task.metadata.name}' does not declare the columns {unknown}. It has "
                f"{sorted(col_modalities)}."
            )
        col_modalities = {column: col_modalities[column] for column in columns}
    if not col_modalities:
        raise NotImplementedError(
            f"`{filter_name}` does not know which columns of '{task.metadata.name}' hold its content. Please "
            "open an issue at https://github.com/embeddings-benchmark/mteb/issues so the task can declare them."
        )
    return col_modalities


def filter_task_rows(
    task: AbsTask,
    keep_fn: KeepIndicesFn,
    *,
    filter_name: str,
    normalize: TextNormalization = "strip",
    remap_duplicates: bool = False,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> AbsTask:
    """Apply `keep_fn` to every selected split of `task`, in place.

    Args:
        task: The task to filter. Its data is loaded first if it is not loaded yet.
        keep_fn: Decides which rows to keep.
        filter_name: The name of the calling filter, used in messages.
        normalize: How to normalize text before comparing it.
        remap_duplicates: Retrieval only; see `filter_retrieval_split`.
        columns: The columns to compare. Defaults to every content column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading and filtering the dataset.

    Returns:
        The task, so that filters can be chained.

    Raises:
        NotImplementedError: If the task does not declare its content columns.
        ValueError: If `columns`, `splits` or `subsets` select none of the task's data.
    """
    if isinstance(task, AbsTaskAggregate):
        # an aggregate task holds no data of its own
        for sub_task in task.tasks:
            filter_task_rows(
                sub_task,
                keep_fn,
                filter_name=filter_name,
                normalize=normalize,
                remap_duplicates=remap_duplicates,
                columns=columns,
                splits=splits,
                subsets=subsets,
                num_proc=num_proc,
            )
        return task

    if not task.data_loaded:
        task.load_data(num_proc=num_proc)

    col_modalities = _resolve_columns(task, filter_name, columns)
    is_retrieval = isinstance(task, AbsTaskRetrieval)
    available: Mapping[str, Any] = cast(
        "Mapping[str, Any]", task.dataset if is_retrieval else datasets_by_subset(task)
    )

    n_removed = 0
    n_filtered_splits = 0
    for subset, splits_data in available.items():
        if subsets is not None and subset not in subsets:
            continue
        for split in list(splits_data.keys()):
            if splits is not None and split not in splits:
                continue
            if is_retrieval:
                splits_data[split], removed = filter_retrieval_split(
                    splits_data[split],
                    keep_fn,
                    col_modalities,
                    normalize=normalize,
                    remap_duplicates=remap_duplicates,
                    num_proc=num_proc,
                )
            else:
                splits_data[split], removed = apply_row_filter(
                    splits_data[split],
                    col_modalities,
                    keep_fn,
                    normalize=normalize,
                    num_proc=num_proc,
                )
            n_removed += removed
            n_filtered_splits += 1

    if n_filtered_splits == 0:
        raise ValueError(_no_split_matched_message(task.metadata.name, available))

    if n_removed:
        mark_data_modified(task)
        warn_about_unusable_data(task)

    logger.info(
        f"`{filter_name}` removed {n_removed} samples from '{task.metadata.name}' "
        f"(columns={sorted(col_modalities)})."
    )
    return task
