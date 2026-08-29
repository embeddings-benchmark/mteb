"""The filters of `mteb.quality`, and the machinery that applies them to a task.

The primitives at the top work on a single `datasets.Dataset` and know nothing about task types: the caller
supplies the columns to compare and a `KeepIndicesFn` deciding which rows to keep. `_filter_task_rows` then walks
a task's subsets and splits, dispatching to `_classification` and `_retrieval` for the parts that differ per task
type. The public filters are at the bottom.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import re
from collections.abc import (
    Callable,
    Iterable,
    Mapping,  # noqa: TC003
)
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

from datasets import Dataset, DatasetDict

from mteb._content_hashes import MODALITY_HASH_FNS
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.retrieval import AbsTaskRetrieval

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from mteb.abstasks.abstask import AbsTask
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import HFSubset, Modalities

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="AbsTask")

Normalization = Callable[[str], str]
"""Rewrites a text into the form a filter compares it in.

Only text is normalized. Images, audio and video are compared by an exact hash of their content, so a re-encoded
or rescaled copy of a sample does not currently match the original.
"""

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)


def strip_whitespace(text: str) -> str:
    """Ignore surrounding whitespace, so that texts must otherwise be identical to match. The default."""
    return text.strip()


def casefold_text(text: str) -> str:
    """Also ignore case, so that `"Hello"` and `"hello"` match."""
    return text.strip().casefold()


def alphanumeric_text(text: str) -> str:
    """Also ignore punctuation and repeated whitespace, so that `"e-mail"` and `"email"` match.

    Punctuation is dropped rather than replaced by a space, so word boundaries still tell texts apart: `"e mail"`
    does not match `"email"`. This is the most aggressive of the three and can merge texts a reader would
    distinguish, e.g. source code or anything where punctuation carries meaning.
    """
    return " ".join(_PUNCTUATION.sub("", text.casefold()).split())


KeepIndicesFn = Callable[[Iterable[tuple[str, ...]]], list[int]]
"""Given the comparable content of each row, return the (ascending) indices of the rows to keep.

A row arrives as one tuple holding the content of each compared column. The rows are passed as a lazy iterable and
may only be consumed once, so that filtering a large corpus does not require holding all of its content in memory
at the same time.
"""


@dataclass(frozen=True)
class _Filter:
    """What a filter removes, and what that means for the relevance judgements of a retrieval task.

    Grouping these keeps them from drifting apart: `removes_duplicates` is only sound because `keep` drops a row
    for being identical to a kept one, which is what lets a retrieval task hand the removed row's judgements over.
    """

    name: str
    keep: KeepIndicesFn
    removes_duplicates: bool = False


SUPPORTED_MODALITIES: frozenset[str] = frozenset(MODALITY_HASH_FNS)
"""The modalities a filter can compare, i.e. those the descriptive statistics know how to hash."""


def _normalize(value: Any, normalization: Normalization) -> str:
    """Apply `normalization`, treating a missing text as an empty one."""
    return normalization(value) if isinstance(value, str) else ""


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
    normalization: Normalization = strip_whitespace,
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
            per_column.append(
                _normalize(value, normalization) for value in dataset[column]
            )
        else:
            per_column.append(
                MODALITY_HASH_FNS[modality](dataset[column], max_workers=num_proc)
            )
    rows = zip(*per_column, strict=True)

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
    normalization: Normalization,
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
        tuple(_normalize(example[column][i], normalization) for column in columns)
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
    normalization: Normalization = strip_whitespace,
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
        normalization: How to rewrite text before comparing it.
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
                "normalization": normalization,
            },
            num_proc=num_proc,
        )
    else:
        rows = _iter_row_content(
            dataset,
            col_modalities,
            normalization=normalization,
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


_APPLIED_FILTERS = re.compile(r"^(?P<base>.*?) \((?P<filters>[^()]*)\)$")


def _derived_task_name(name: str, filter_name: str) -> str:
    """The name a task takes once `filter_name` has been applied to it.

    Cleaning produces a different task, so it gets an id of its own rather than reusing the published one:
    `MassiveIntentClassification` becomes `MassiveIntentClassification (remove_duplicates)`. A second filter
    extends the list rather than nesting, giving `MassiveIntentClassification (remove_duplicates, filter_short)`.
    """
    applied_to = _APPLIED_FILTERS.match(name)
    if applied_to is None:
        return f"{name} ({filter_name})"

    applied = [applied.strip() for applied in applied_to["filters"].split(",")]
    if filter_name not in applied:
        applied.append(filter_name)
    return f"{applied_to['base']} ({', '.join(applied)})"


def _rename_as_cleaned(task: AbsTask, original: TaskMetadata, filter_name: str) -> None:
    """Give `task` a metadata of its own, named after the filter that produced it.

    `metadata` is a class attribute shared by every instance of a task, so this assigns an instance attribute that
    shadows it, leaving the published task and its other instances alone.
    """
    base = _APPLIED_FILTERS.match(original.name)
    task.metadata = original.model_copy(
        update={
            "name": _derived_task_name(original.name, filter_name),
            "adapted_from": [base["base"] if base else original.name],
        }
    )


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


def _split_containers(task: AbsTask) -> tuple[Mapping[str, Any], bool]:
    """The task's `{subset: {split: data}}` mapping, and whether it was stored without the subset level."""
    if isinstance(task, AbsTaskRetrieval):
        return cast("Mapping[str, Any]", task.dataset), False
    by_subset = _datasets_by_subset(task)
    flat = isinstance(next(iter(cast("Any", task.dataset).values()), None), Dataset)
    return by_subset, flat


def _filter_task_rows(
    task: T,
    filter_: _Filter,
    *,
    normalization: Normalization = strip_whitespace,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Apply `filter_` to every selected split of `task`, returning a cleaned copy.

    The task passed in is never changed. Its data is loaded if needed, and the returned copy holds new containers
    for the splits that were filtered, so the two share nothing that either could mutate.

    Args:
        task: The task to filter.
        filter_: What to remove.
        normalization: How to rewrite text before comparing it.
        columns: The columns to compare. Defaults to every content column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading and filtering the dataset.

    Returns:
        A copy of the task holding the filtered data.

    Raises:
        ValueError: If `columns` names a column the task does not declare, or if `splits` and `subsets` together
            match none of the task's splits.
    """
    from ._retrieval import _filter_retrieval_split

    if isinstance(task, AbsTaskAggregate):
        # an aggregate task holds no data of its own, only the tasks it aggregates
        sub_tasks = [
            _filter_task_rows(
                sub_task,
                filter_,
                normalization=normalization,
                columns=columns,
                splits=splits,
                subsets=subsets,
                num_proc=num_proc,
            )
            for sub_task in task.tasks
        ]
        cleaned_aggregate = copy.copy(task)
        cleaned_aggregate.tasks = sub_tasks
        cleaned_aggregate.taskname_to_task = {t.metadata.name: t for t in sub_tasks}
        if any(
            cleaned_sub.metadata.name != original.metadata.name
            for cleaned_sub, original in zip(sub_tasks, task.tasks, strict=True)
        ):
            _rename_as_cleaned(cleaned_aggregate, task.metadata, filter_.name)
        return cleaned_aggregate

    if not task.data_loaded:
        task.load_data(num_proc=num_proc)

    col_modalities = _resolve_columns(task, filter_.name, columns)
    symmetric_sides = _resolve_symmetric_sides(task, col_modalities)
    is_retrieval = isinstance(task, AbsTaskRetrieval)
    available, flat = _split_containers(task)

    n_removed = 0
    n_filtered_splits = 0
    by_subset: dict[str, Any] = {}
    for subset, splits_data in available.items():
        new_splits = dict(splits_data)
        for split in splits_data:
            if subsets is not None and subset not in subsets:
                continue
            if splits is not None and split not in splits:
                continue
            if is_retrieval:
                new_splits[split], removed = _filter_retrieval_split(
                    splits_data[split],
                    filter_.keep,
                    col_modalities,
                    normalization=normalization,
                    remap_duplicates=filter_.removes_duplicates,
                    num_proc=num_proc,
                )
            else:
                new_splits[split], removed = _apply_row_filter(
                    splits_data[split],
                    col_modalities,
                    filter_.keep,
                    normalization=normalization,
                    num_proc=num_proc,
                    symmetric_sides=symmetric_sides,
                )
            n_removed += removed
            n_filtered_splits += 1
        by_subset[subset] = new_splits if is_retrieval else DatasetDict(new_splits)

    if n_filtered_splits == 0:
        raise ValueError(_no_split_matched_message(task.metadata.name, available))

    cleaned = copy.copy(task)
    cleaned.dataset = by_subset["default"] if flat else by_subset
    if n_removed:
        _rename_as_cleaned(cleaned, task.metadata, filter_.name)
        _warn(
            f"`{filter_.name}` removed {n_removed} samples from '{task.metadata.name}' "
            f"(columns={sorted(col_modalities)}). The cleaned task is '{cleaned.metadata.name}', and its scores "
            f"are not comparable to results on '{task.metadata.name}'."
        )
    else:
        logger.info(
            f"`{filter_.name}` removed nothing from '{task.metadata.name}' "
            f"(columns={sorted(col_modalities)})."
        )
    return cleaned


def remove_duplicates(
    task: T,
    *,
    normalization: Normalization = strip_whitespace,
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
    document, and any query left without one afterwards is dropped, as it cannot be scored.

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
        ValueError: If `columns` names a column the task does not declare, or if `splits` and `subsets` together
            match none of the task's splits.

    Examples:
        >>> import mteb
        >>> from mteb.quality import remove_duplicates
        >>> task = mteb.get_task("MassiveIntentClassification")
        >>> cleaned = remove_duplicates(task)
        >>> # ignore case too, so that "Wake me up!" and "wake me up!" are duplicates
        >>> cleaned = remove_duplicates(task, normalization=lambda t: t.strip().casefold())
    """
    return _filter_task_rows(
        task,
        _Filter("remove_duplicates", _keep_first_occurrence, removes_duplicates=True),
        normalization=normalization,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
