"""The filters of `mteb.data_cleaning`, and the machinery that applies them to a task.

The primitives at the top work on a single `datasets.Dataset` and know nothing about task types: the caller
supplies the columns to compare and a `KeepRowsFn` deciding which rows to keep. `_filter_task_rows` then walks
a task's subsets and splits, dispatching to `_retrieval` for the parts that differ per task type. The public filters are at the bottom.
"""

from __future__ import annotations

import copy
import functools
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
from mteb._set_seed import _set_seed
from mteb.abstasks._statistics_calculation import (
    _audio_duration_seconds,
    _video_duration_seconds,
)
from mteb.abstasks.aggregated_task import AbsTaskAggregate
from mteb.abstasks.retrieval import AbsTaskRetrieval
from mteb.abstasks.sts import AbsTaskSTS

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from PIL import Image

    from mteb.abstasks.abstask import AbsTask
    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import HFSubset, Modalities

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="AbsTask")

Normalization = Callable[[str], str]
"""Rewrites a text into the form a filter compares it in.

The default only strips surrounding whitespace. Pass your own to ignore more, e.g. case or punctuation.

Only text is normalized. Images, audio and video are compared by an exact hash of their content, so a re-encoded
or rescaled copy of a sample does not currently match the original.
"""


def _strip_whitespace(text: str) -> str:
    """Ignore surrounding whitespace, so that texts must otherwise be identical to match. The default."""
    return text.strip()


KeepRowsFn = Callable[[Iterable[tuple[Any, ...]]], list[int]]
"""Given the content of each row, return the (ascending) indices of the rows to keep.

A row arrives as one tuple holding the content of each compared column: text as a normalized string, and images,
audio and video as a hash of their content, or as they are for a filter that measures rows rather than comparing
them. The rows are passed as a lazy iterable and may only be consumed once, so that filtering a large corpus does
not require holding all of its content in memory at the same time.
"""


@dataclass(frozen=True)
class _Filter:
    """What a filter removes, and what that means for the relevance judgements of a retrieval task.

    Grouping these keeps them from drifting apart: `removes_duplicates` is only sound because `keep_fn` drops a row
    for being identical to a kept one, which is what lets a retrieval task hand the removed row's judgements over.

    Attributes:
        name: The filter's name, used in messages and in the name of the cleaned task.
        keep_fn: Decides which rows survive, given the comparable content of each.
        removes_duplicates: Whether a row is removed for being identical to a kept one. When it is, a retrieval
            task moves the removed row's relevance judgements to the row it duplicated, so that deduplication
            costs no query its positives. A filter that removes rows on their own merit must leave this False.
        modalities: The modalities the filter applies to, or None for every one. The content columns of any
            other modality are left out of what `keep_fn` sees.
        compares_rows: Whether `keep_fn` compares rows with each other, which it does by the hash of their
            images, audio and video. A filter that measures each row on its own sees them as they are instead.
    """

    name: str
    keep_fn: KeepRowsFn
    removes_duplicates: bool = False
    modalities: frozenset[Modalities] | None = None
    compares_rows: bool = True


_SUPPORTED_MODALITIES: frozenset[str] = frozenset(MODALITY_HASH_FNS)
"""The modalities a filter can compare, i.e. those the descriptive statistics know how to hash."""


def _normalize(value: object, normalization: Normalization) -> str:
    """Apply `normalization` to a value read from a column, treating a missing text as an empty one."""
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


def _keep_at_least(
    rows: Iterable[tuple[Any, ...]],
    *,
    minimum: float,
    measure_fn: Callable[[Any], float | None],
) -> list[int]:
    """Keep the rows whose values all measure at least `minimum`.

    Args:
        rows: The content of each row, one tuple per row with one entry per compared column.
        minimum: The smallest size a value may have.
        measure_fn: The size of a value, or None if it cannot be told, in which case the value is kept.

    Returns:
        The indices of the rows without a value smaller than `minimum`.
    """

    def large_enough(value: object) -> bool:
        size = measure_fn(value)
        return size is None or size >= minimum

    return [i for i, row in enumerate(rows) if all(map(large_enough, row))]


def _content_readers(
    dataset: Dataset,
    col_modalities: Mapping[str, Modalities],
    *,
    normalization: Normalization,
    hash_non_text: bool,
    num_proc: int | None,
) -> list[Callable[[], Iterable[Any]]]:
    """One reader per compared column, each returning that column's content when called.

    Hashing images, audio or video is expensive, so it happens once here and the result is reused every time the
    rows are read. Text, and non-text content that is not hashed, stays lazy, so a large corpus is never held in
    memory at once.
    """
    readers: list[Callable[[], Iterable[Any]]] = []
    for column, modality in col_modalities.items():
        if modality == "text":

            def _read_text(column: str = column) -> Iterable[str]:
                return (_normalize(value, normalization) for value in dataset[column])

            readers.append(_read_text)
        elif hash_non_text:
            hashes = MODALITY_HASH_FNS[modality](dataset[column], max_workers=num_proc)
            readers.append(lambda hashes=hashes: hashes)  # type: ignore[misc]
        else:
            readers.append(lambda column=column: dataset[column])  # type: ignore[misc]
    return readers


def _iter_row_content(
    readers: Sequence[Callable[[], Iterable[Any]]],
    *,
    columns: Sequence[str] = (),
    symmetric_sides: tuple[list[str], list[str]] | None = None,
) -> Iterator[tuple[Any, ...]]:
    """Iterate the comparable content of each row, one entry per reader.

    When `symmetric_sides` names the two sides of a symmetric task, they are ordered within each row, so that a
    pair and its swap compare equal.
    """
    rows = zip(*(read() for read in readers), strict=True)

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
    """The two sides of a task that mean the same thing when swapped, if they cover the compared columns.

    STS is the only symmetric task type: a similarity does not depend on which sentence comes first, which is why
    it already passes `symmetric=True` when counting unique pairs for its descriptive statistics. Narrowing the
    comparison with `columns=` can leave a side partly selected, in which case swapping them is no longer
    meaningful and the comparison stays order-sensitive.
    """
    if not isinstance(task, AbsTaskSTS):
        return None
    left, right = (
        [column] if isinstance(column, str) else list(column)
        for column in task.column_names
    )
    if set(left) | set(right) != set(col_modalities):
        return None
    return left, right


def _is_grouped(dataset: Dataset, columns: Sequence[str]) -> bool:
    """Whether each row holds a list of values (e.g. the sentences of a cluster) rather than a single one."""
    return isinstance(dataset[0][columns[0]], list)


def _filter_within_row(
    example: dict[str, Any],
    columns: Sequence[str],
    keep_fn: KeepRowsFn,
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
    filter_: _Filter,
    *,
    normalization: Normalization = _strip_whitespace,
    num_proc: int | None = None,
    symmetric_sides: tuple[list[str], list[str]] | None = None,
) -> tuple[Dataset, int]:
    """Filter `dataset` down to the rows that `filter_` keeps.

    For a regular dataset this drops whole rows. For a grouped dataset -- one where each row holds a list of values,
    as clustering tasks do -- the filter is applied within each row instead, and the parallel columns of that row
    (e.g. the labels) are filtered along with it.

    Args:
        dataset: The dataset to filter.
        col_modalities: The columns to compare, mapped to the modality of their content.
        filter_: Decides which rows to keep.
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
                "keep_fn": filter_.keep_fn,
                "normalization": normalization,
            },
            num_proc=num_proc,
        )
    else:
        readers = _content_readers(
            dataset,
            col_modalities,
            normalization=normalization,
            hash_non_text=filter_.compares_rows,
            num_proc=num_proc,
        )
        rows = _iter_row_content(
            readers, columns=columns, symmetric_sides=symmetric_sides
        )
        filtered = dataset.select(filter_.keep_fn(rows))

    return filtered, before - _count_values(filtered, columns[0], grouped)


_APPLIED_FILTERS = re.compile(r"^(?P<base>.*?) \((?P<filters>[^()]*)\)$")


def _independent_copy(task: T) -> T:
    """A copy of `task` that shares no mutable state with it.

    `copy.copy` would leave the two pointing at the same list of subsets and the same random number generators, so
    working with one task could reach into the other: narrowing the copy's languages in place would narrow the
    original's, and drawing from one generator would advance the other's sampling.
    """
    cleaned = copy.copy(task)
    cleaned.hf_subsets = list(task.hf_subsets)
    cleaned.rng_state, cleaned.np_rng = _set_seed(task.seed)
    return cleaned


def _derived_task_name(name: str, filter_name: str) -> str:
    """The name a task takes once `filter_name` has been applied to it.

    Cleaning produces a different task, so it gets an id of its own rather than reusing the published one:
    `MassiveIntentClassification` becomes `MassiveIntentClassification (remove_duplicates)`. A second filter
    extends the list rather than nesting, giving
    `MassiveIntentClassification (remove_duplicates, remove_short_texts)`.
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
    task: AbsTask, filter_: _Filter, columns: Sequence[str] | None
) -> dict[str, Modalities]:
    """The columns a filter should compare, mapped to the modality of their content."""
    col_modalities = task._get_content_columns()
    if not col_modalities:
        raise NotImplementedError(
            f"`{filter_.name}` does not know which columns of '{task.metadata.name}' hold its content. Please "
            "open an issue at https://github.com/embeddings-benchmark/mteb/issues so the task can declare them."
        )

    if filter_.modalities is not None:
        col_modalities = {
            column: modality
            for column, modality in col_modalities.items()
            if modality in filter_.modalities
        }
        if not col_modalities:
            raise ValueError(
                f"`{filter_.name}` only applies to {sorted(filter_.modalities)} content, which "
                f"'{task.metadata.name}' does not have."
            )

    if columns is not None:
        # a column the task does not declare, or that the filter does not apply to, raises a KeyError naming it
        col_modalities = {column: col_modalities[column] for column in columns}

    unsupported = sorted(set(col_modalities.values()) - _SUPPORTED_MODALITIES)
    if unsupported:
        raise NotImplementedError(
            f"`{filter_.name}` cannot compare the {unsupported} content of '{task.metadata.name}'. Supported "
            f"modalities are {sorted(_SUPPORTED_MODALITIES)}."
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
    normalization: Normalization = _strip_whitespace,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Apply `filter_` to every selected split of `task`, returning a cleaned copy.

    The task passed in is never changed, not even by loading its data: the copy is made first and the data is
    loaded onto that. The copy holds new containers for any filtered splits; unfiltered splits and subsets may
    still be shared with the input task.


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
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If `task` holds none of the content `filter_` applies to, or if `splits` and `subsets`
            together match none of the task's splits.
        KeyError: If `columns` names a column the task does not declare.
    """
    from ._retrieval import _filter_retrieval_split

    if isinstance(task, AbsTaskAggregate):
        raise NotImplementedError(
            f"'{task.metadata.name}' aggregates other tasks and holds no data of its own. Filter the tasks it "
            "aggregates instead, via its `tasks` attribute."
        )

    # copy before loading, so that a task whose data is not loaded yet is left that way
    cleaned = _independent_copy(task)
    original = cleaned.metadata
    if not cleaned.data_loaded:
        cleaned.load_data(num_proc=num_proc)
    if isinstance(cleaned, AbsTaskRetrieval):
        # some tasks still load the older corpus/queries layout, which evaluation converts first as well
        cleaned.convert_v1_dataset_format_to_v2(num_proc=num_proc)

    col_modalities = _resolve_columns(cleaned, filter_, columns)
    # a pair and its swap only need to meet when rows are compared with each other
    symmetric_sides = (
        _resolve_symmetric_sides(cleaned, col_modalities)
        if filter_.compares_rows
        else None
    )
    is_retrieval = isinstance(cleaned, AbsTaskRetrieval)
    available, flat = _split_containers(cleaned)

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
                    filter_,
                    col_modalities,
                    original,
                    normalization=normalization,
                    num_proc=num_proc,
                )
            else:
                new_splits[split], removed = _apply_row_filter(
                    splits_data[split],
                    col_modalities,
                    filter_,
                    normalization=normalization,
                    num_proc=num_proc,
                    symmetric_sides=symmetric_sides,
                )
            n_removed += removed
            n_filtered_splits += 1
        by_subset[subset] = new_splits if is_retrieval else DatasetDict(new_splits)

    if n_filtered_splits == 0:
        raise ValueError(_no_split_matched_message(original.name, available))

    cleaned.dataset = by_subset["default"] if flat else by_subset
    if n_removed:
        _rename_as_cleaned(cleaned, original, filter_.name)
        logger.warning(
            f"`{filter_.name}` removed {n_removed} samples from '{original.name}' "
            f"(columns={sorted(col_modalities)}). The cleaned task is '{cleaned.metadata.name}', and its scores "
            f"are not comparable to results on '{original.name}'."
        )
    else:
        logger.info(
            f"`{filter_.name}` removed nothing from '{original.name}' "
            f"(columns={sorted(col_modalities)})."
        )
    return cleaned


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
        _Filter("remove_duplicates", _keep_first_occurrence, removes_duplicates=True),
        normalization=normalization,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def _remove_small(
    task: T,
    name: str,
    modality: Modalities,
    minimum: float,
    measure_fn: Callable[[Any], float | None],
    *,
    columns: Sequence[str] | None,
    splits: Sequence[str] | None,
    subsets: Sequence[HFSubset] | None,
    num_proc: int | None,
) -> T:
    """Remove the samples holding a `modality` value that `measure_fn` finds smaller than `minimum`."""
    return _filter_task_rows(
        task,
        _Filter(
            name,
            functools.partial(_keep_at_least, minimum=minimum, measure_fn=measure_fn),
            modalities=frozenset({modality}),
            compares_rows=False,
        ),
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def remove_short_texts(
    task: T,
    *,
    min_length: int,
    length_fn: Callable[[str], int] = len,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove samples with a text shorter than `min_length`, which includes empty and whitespace-only texts.

    A sample is removed when any of its texts is too short, e.g. either sentence of a pair. Texts are counted in
    characters, ignoring surrounding whitespace, and a missing text counts as empty. A retrieval document is measured
    on its title and text together and a query on its text and instruction, as they are encoded that way when
    evaluated: a removed document takes its relevance judgements with it, and a query left without one is dropped.

    Only text is measured, so images, audio and video are kept whatever their size, as is a retrieval document or
    query that combines text with one of them, e.g. a page whose image carries the content its text lacks.

    Args:
        task: The task to filter. It is not modified.
        min_length: The shortest length a text may have. `1` removes only empty and whitespace-only texts.
        length_fn: How to measure a text. Defaults to its number of characters.
        columns: The text columns to measure. Defaults to every text column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading and filtering the dataset.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `MassiveIntentClassification (remove_short_texts)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If `task` has no text, or if `splits` and `subsets` together match none of its splits.
        KeyError: If `columns` names a column that is not a text column of the task.

    Examples:
        >>> import mteb
        >>> from mteb.data_cleaning import remove_short_texts
        >>> task = mteb.get_task("MassiveIntentClassification")
        >>> cleaned = remove_short_texts(task, min_length=1)  # empty and whitespace-only texts
        >>> cleaned = remove_short_texts(task, min_length=3, length_fn=lambda text: len(text.split()))  # < 3 words
    """
    return _remove_small(
        task,
        "remove_short_texts",
        "text",
        min_length,
        length_fn,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def _shorter_side(image: Image.Image) -> int:
    """The shorter of an image's width and height, so that a thin image counts as small. The default."""
    return min(image.size)


def remove_small_images(
    task: T,
    *,
    min_size: int,
    size_fn: Callable[[Image.Image], float] = _shorter_side,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove samples with an image smaller than `min_size`, e.g. a 1x1 placeholder.

    A sample is removed when any of its images is too small. An image is as small as its shorter side by default, so
    `min_size` is the width and the height it must reach, as the `min_image_width` and `min_image_height` of the
    task's descriptive statistics report them. Only images are measured, so the texts of a sample are kept whatever
    their length.

    Args:
        task: The task to filter. It is not modified.
        min_size: The smallest size, by `size_fn`, an image may have.
        size_fn: How to measure an image. Defaults to its shorter side in pixels; `lambda image: image.width *
            image.height` measures its area instead, in which case `min_size` is a number of pixels.
        columns: The image columns to measure. Defaults to every image column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `ROxfordEasyI2IRetrieval (remove_small_images)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If `task` has no images, or if `splits` and `subsets` together match none of its splits.
        KeyError: If `columns` names a column that is not an image column of the task.
    """
    return _remove_small(
        task,
        "remove_small_images",
        "image",
        min_size,
        size_fn,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def remove_short_audio(
    task: T,
    *,
    min_seconds: float,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove samples with an audio clip shorter than `min_seconds`.

    A sample is removed when any of its clips is too short, measured as the `min_duration_seconds` of the task's
    descriptive statistics measures it. Only audio is measured, so the texts of a sample are kept whatever their
    length.

    Args:
        task: The task to filter. It is not modified.
        min_seconds: The shortest duration, in seconds, an audio clip may have.
        columns: The audio columns to measure. Defaults to every audio column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `BeijingOpera (remove_short_audio)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If `task` has no audio, or if `splits` and `subsets` together match none of its splits.
        KeyError: If `columns` names a column that is not an audio column of the task.
    """
    return _remove_small(
        task,
        "remove_short_audio",
        "audio",
        min_seconds,
        _audio_duration_seconds,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def remove_short_videos(
    task: T,
    *,
    min_seconds: float,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove samples with a video shorter than `min_seconds`. A video whose duration is unknown is kept.

    A sample is removed when any of its videos is too short, measured as the `min_duration_seconds` of the task's
    descriptive statistics measures it. Only video is measured, so the texts of a sample are kept whatever their
    length.

    Args:
        task: The task to filter. It is not modified.
        min_seconds: The shortest duration, in seconds, a video may have.
        columns: The video columns to measure. Defaults to every video column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `CoVRRVT2VRetrieval (remove_short_videos)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If `task` has no videos, or if `splits` and `subsets` together match none of its splits.
        KeyError: If `columns` names a column that is not a video column of the task.
    """
    return _remove_small(
        task,
        "remove_short_videos",
        "video",
        min_seconds,
        _video_duration_seconds,
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
