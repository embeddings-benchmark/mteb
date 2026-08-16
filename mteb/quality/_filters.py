"""The filters of `mteb.quality`, and the machinery that applies them to a task.

The primitives at the top work on a single `datasets.Dataset` and know nothing about task types: the caller
supplies the columns to compare and a `KeepIndicesFn` deciding which rows to keep. `_filter_task_rows` then walks
a task's subsets and splits, dispatching to `_classification` and `_retrieval` for the parts that differ per task
type. The public filters are at the bottom.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import (
    Callable,
    Iterable,
    Mapping,  # noqa: TC003
)
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from datasets import Dataset

from mteb._content_hashes import MODALITY_HASH_FNS
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.classification import AbsTaskClassification
from mteb.abstasks.retrieval import AbsTaskRetrieval

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from datasets import DatasetDict

    from mteb.abstasks.abstask import AbsTask
    from mteb.types import HFSubset, Modalities

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="AbsTask")

TextNormalization = Literal["strip", "casefold", "alphanumeric"]
"""How much of a difference between two documents to ignore when deciding whether they are duplicates.

- `"strip"`: only ignore surrounding whitespace, so the texts must otherwise be identical.
- `"casefold"`: also ignore case, so `"Hello"` and `"hello"` are duplicates.
- `"alphanumeric"`: also ignore punctuation and repeated whitespace, so `"hello, world!"` and `"Hello world"` are
    duplicates, as are `"e-mail"` and `"email"`. This is the most aggressive option and can merge documents that a
    reader would tell apart, e.g. source code or texts where punctuation carries the meaning. Word boundaries are
    still significant, so `"e mail"` remains distinct from `"email"`.

It only applies to text. Images, audio and video are always compared by the hash of their content.
"""

KeepIndicesFn = Callable[[Iterable[tuple[str, ...]]], list[int]]
"""Given the comparable content of each row, return the (ascending) indices of the rows to keep.

A row arrives as one tuple holding the content of each compared column. The rows are passed as a lazy iterable and
may only be consumed once, so that filtering a large corpus does not require holding all of its content in memory
at the same time.
"""

SUPPORTED_MODALITIES: frozenset[str] = frozenset(MODALITY_HASH_FNS)
"""The modalities a filter can compare, i.e. those the descriptive statistics know how to hash."""

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_MIN_TRAIN_EXAMPLES_PER_LABEL = 2
"""Stratified subsampling and the train/test split both need at least this many examples of each label."""


def _normalize(value: Any, normalize: TextNormalization = "strip") -> str:
    """Bring a text into the form that `normalize` compares, treating a missing text as an empty one."""
    if not isinstance(value, str):
        return ""
    text = value.strip()
    if normalize == "strip":
        return text
    text = text.casefold()
    if normalize == "casefold":
        return text
    # drop the punctuation instead of replacing it, so that "e-mail" and "email" compare equal
    return " ".join(_PUNCTUATION.sub("", text).split())


def _row_key(row: tuple[str, ...]) -> bytes:
    """A compact key identifying a row by the comparable content of its columns.

    Hashing rather than keeping the content itself makes the memory used while deduplicating proportional to the
    number of rows instead of to the size of the corpus, which matters for the larger retrieval datasets. Each
    value is length-prefixed so that a row cannot collide with a differently split one, e.g. `("a", "b")` and
    `("ab", "")`.
    """
    digest = hashlib.blake2b(digest_size=16)
    for value in row:
        encoded = value.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.digest()


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


def _iter_row_content(
    dataset: Dataset,
    col_modalities: Mapping[str, Modalities],
    *,
    normalize: TextNormalization = "strip",
    num_proc: int | None = None,
    symmetric_sides: tuple[list[str], list[str]] | None = None,
) -> Iterator[tuple[str, ...]]:
    """Iterate the comparable content of each row, one entry per column of `col_modalities`.

    Text is compared by its normalized value; every other modality is compared by a hash of its content, using the
    same hash functions as the descriptive statistics. Text columns stay lazy so that the memory needed to compare
    a large text corpus does not grow with its size; the other modalities have to be decoded to be hashed.

    When `symmetric_sides` names the two sides of a symmetric task, they are ordered within each row, so that a
    pair and its swap compare equal.
    """
    columns = list(col_modalities)
    per_column: list[Iterable[str]] = []
    for column, modality in col_modalities.items():
        if modality == "text":
            per_column.append(_normalize(value, normalize) for value in dataset[column])
        else:
            per_column.append(
                MODALITY_HASH_FNS[modality](dataset[column], max_workers=num_proc)
            )
    rows = zip(*per_column)

    if symmetric_sides is None:
        return rows
    left, right = ([columns.index(c) for c in side] for side in symmetric_sides)
    return (
        tuple(
            value
            for side in sorted(
                (tuple(row[i] for i in left), tuple(row[i] for i in right))
            )
            for value in side
        )
        for row in rows
    )


def _resolve_symmetric_sides(
    task: AbsTask, col_modalities: Mapping[str, Modalities]
) -> tuple[list[str], list[str]] | None:
    """The task's symmetric sides, if they exactly cover the columns being compared.

    Narrowing the comparison with `columns=` can leave a side partly selected, in which case swapping the sides is
    no longer meaningful and the comparison stays order-sensitive.
    """
    sides = task._get_symmetric_sides()
    if sides is None or set(sides[0]) | set(sides[1]) != set(col_modalities):
        return None
    return sides


def _is_grouped(dataset: Dataset, columns: Sequence[str]) -> bool:
    """Whether each row holds a list of values (e.g. the sentences of a cluster) rather than a single one."""
    return isinstance(dataset[0][columns[0]], list)


def _filter_within_row(
    example: dict[str, Any],
    columns: Sequence[str],
    keep_fn: KeepIndicesFn,
    normalize: TextNormalization,
) -> dict[str, Any]:
    """Apply `keep_fn` inside a single row of a grouped dataset.

    Every other column of the row that is a list of the same length is filtered alongside the compared columns,
    which keeps parallel columns such as the cluster labels aligned with their texts.
    """
    lengths = {len(example[column]) for column in columns}
    if len(lengths) != 1:
        raise ValueError(
            f"The grouped columns {list(columns)} of a row have differing lengths {sorted(lengths)}, "
            "so they cannot be filtered together."
        )
    n_values = lengths.pop()
    keep = keep_fn(
        tuple(_normalize(example[column][i], normalize) for column in columns)
        for i in range(n_values)
    )
    return {
        column: [value[i] for i in keep]
        if isinstance(value, list) and len(value) == n_values
        else value
        for column, value in example.items()
    }


def _count_values(dataset: Dataset, column: str, grouped: bool) -> int:
    if not grouped:
        return len(dataset)
    return sum(len(values) for values in dataset[column])


def _apply_row_filter(
    dataset: Dataset,
    col_modalities: Mapping[str, Modalities],
    keep_fn: KeepIndicesFn,
    *,
    normalize: TextNormalization = "strip",
    num_proc: int | None = None,
    symmetric_sides: tuple[list[str], list[str]] | None = None,
) -> tuple[Dataset, int]:
    """Filter `dataset` down to the rows that `keep_fn` keeps.

    For a regular dataset this drops whole rows. For a grouped dataset -- one where each row holds a list of values,
    as clustering tasks do -- the filter is applied within each row instead, and the parallel columns of that row
    (e.g. the labels) are filtered along with it.

    Args:
        dataset: The dataset to filter.
        col_modalities: The columns to compare, mapped to the modality of their content.
        keep_fn: Decides which rows to keep.
        normalize: How to normalize text before comparing it.
        num_proc: Number of processes to use for hashing and for filtering a grouped dataset.
        symmetric_sides: The two sides to order within each row, for a task where swapping them means the same.

    Returns:
        The filtered dataset and the number of values that were removed.

    Raises:
        ValueError: If a compared column is missing from the dataset, or if a grouped column is not text.
    """
    columns = list(col_modalities)
    missing = [column for column in columns if column not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Cannot filter on {missing}: the dataset only has the columns {dataset.column_names}."
        )
    if len(dataset) == 0:
        return dataset, 0

    grouped = _is_grouped(dataset, columns)
    before = _count_values(dataset, columns[0], grouped)

    if grouped:
        non_text = sorted(
            column for column, modality in col_modalities.items() if modality != "text"
        )
        if non_text:
            raise ValueError(
                f"The columns {non_text} hold a list per row, which is only supported for text."
            )
        filtered = dataset.map(
            _filter_within_row,
            fn_kwargs={
                "columns": columns,
                "keep_fn": keep_fn,
                "normalize": normalize,
            },
            num_proc=num_proc,
        )
    else:
        rows = _iter_row_content(
            dataset,
            col_modalities,
            normalize=normalize,
            num_proc=num_proc,
            symmetric_sides=symmetric_sides,
        )
        filtered = dataset.select(keep_fn(rows))

    return filtered, before - _count_values(filtered, columns[0], grouped)


def _warn(msg: str) -> None:
    """Report something the caller should know about, without raising.

    These are expected outcomes of filtering rather than misuse of the API, so they are logged rather than raised
    as `UserWarning`s.
    """
    logger.warning(msg)


def _mark_data_modified(task: AbsTask) -> None:
    """Record that the task's data no longer matches the published dataset, reporting it the first time."""
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


def _datasets_by_subset(task: AbsTask) -> dict[HFSubset, DatasetDict]:
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


def _check_unusable_data(task: AbsTask) -> None:
    """Report data that a filter left in a state the evaluators cannot handle."""
    from ._classification import _check_label_distribution
    from ._retrieval import _check_empty_retrieval_splits

    if isinstance(task, AbsTaskRetrieval):
        _check_empty_retrieval_splits(task)
        return

    for subset, dataset_dict in _datasets_by_subset(task).items():
        empty = sorted(split for split, ds in dataset_dict.items() if len(ds) == 0)
        if empty:
            _warn(
                f"Filtering left the splits {empty} of '{task.metadata.name}' (subset '{subset}') empty. "
                "Evaluating them will fail."
            )

    if isinstance(task, AbsTaskClassification):
        _check_label_distribution(
            task, min_examples_per_label=_MIN_TRAIN_EXAMPLES_PER_LABEL
        )


def _resolve_columns(
    task: AbsTask, filter_name: str, columns: Sequence[str] | None
) -> dict[str, Modalities]:
    """The columns a filter should compare, mapped to the modality of their content."""
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

    unsupported = sorted(set(col_modalities.values()) - SUPPORTED_MODALITIES)
    if unsupported:
        raise NotImplementedError(
            f"`{filter_name}` cannot compare the {unsupported} content of '{task.metadata.name}'. Supported "
            f"modalities are {sorted(SUPPORTED_MODALITIES)}."
        )
    return col_modalities


def _filter_task_rows(
    task: T,
    keep_fn: KeepIndicesFn,
    *,
    filter_name: str,
    normalize: TextNormalization = "strip",
    remap_duplicates: bool = False,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Apply `keep_fn` to every selected split of `task`, in place.

    Args:
        task: The task to filter. Its data is loaded first if it is not loaded yet.
        keep_fn: Decides which rows to keep.
        filter_name: The name of the calling filter, used in messages.
        normalize: How to normalize text before comparing it.
        remap_duplicates: Retrieval only; see `_filter_retrieval_split`.
        columns: The columns to compare. Defaults to every content column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading and filtering the dataset.

    Returns:
        The task, so that filters can be chained.

    Raises:
        NotImplementedError: If the task does not declare its content columns, or holds a modality that cannot be
            compared.
        ValueError: If `columns`, `splits` or `subsets` select none of the task's data.
    """
    from ._retrieval import _filter_retrieval_split

    if isinstance(task, AbsTaskAggregate):
        # an aggregate task holds no data of its own
        for sub_task in task.tasks:
            _filter_task_rows(
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
    symmetric_sides = _resolve_symmetric_sides(task, col_modalities)
    is_retrieval = isinstance(task, AbsTaskRetrieval)
    available: Mapping[str, Any] = cast(
        "Mapping[str, Any]", task.dataset if is_retrieval else _datasets_by_subset(task)
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
                splits_data[split], removed = _filter_retrieval_split(
                    splits_data[split],
                    keep_fn,
                    col_modalities,
                    normalize=normalize,
                    remap_duplicates=remap_duplicates,
                    num_proc=num_proc,
                )
            else:
                splits_data[split], removed = _apply_row_filter(
                    splits_data[split],
                    col_modalities,
                    keep_fn,
                    normalize=normalize,
                    num_proc=num_proc,
                    symmetric_sides=symmetric_sides,
                )
            n_removed += removed
            n_filtered_splits += 1

    if n_filtered_splits == 0:
        raise ValueError(_no_split_matched_message(task.metadata.name, available))

    if n_removed:
        _mark_data_modified(task)
        _check_unusable_data(task)

    logger.info(
        f"`{filter_name}` removed {n_removed} samples from '{task.metadata.name}' "
        f"(columns={sorted(col_modalities)})."
    )
    return task


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
    marked with `data_modified` and can no longer be evaluated with [`mteb.evaluate`][mteb.evaluate].

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
        NotImplementedError: If the task does not declare which of its columns hold content, or holds a modality
            that cannot be compared.
        ValueError: If `columns`, `splits` or `subsets` select none of the task's data.

    Examples:
        >>> import mteb
        >>> from mteb.quality import remove_duplicates
        >>> task = mteb.get_task("MassiveIntentClassification")
        >>> task = remove_duplicates(task)
        >>> # also treat "Wake me up!" and "wake  me  up" as duplicates
        >>> task = remove_duplicates(task, normalize="alphanumeric")
    """
    return _filter_task_rows(
        task,
        _keep_first_occurrence,
        filter_name="remove_duplicates",
        normalize=normalize,
        remap_duplicates=True,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
