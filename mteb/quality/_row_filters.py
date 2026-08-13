"""Column-aware row filters, the building blocks the public `mteb.quality` filters are assembled from.

The functions here work on a single `datasets.Dataset` and know nothing about task types: the caller supplies the
columns to compare and a `KeepIndicesFn` deciding which rows to keep.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal

from mteb.abstasks._statistics_calculation import _HASH_FN

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from datasets import Dataset

    from mteb.types import Modalities

logger = logging.getLogger(__name__)

TextLengthUnit = Literal["characters", "words"]
"""The unit used when measuring the length of a document."""

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
"""Given the comparison key of each row, return the (ascending) indices of the rows to keep.

The keys are passed as a lazy iterable and may only be consumed once, so that filtering a large corpus does not
require holding all of its content in memory at the same time.
"""

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)


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


def row_key(keys: tuple[str, ...]) -> bytes:
    """A compact key identifying a row by the comparison keys of its columns.

    Hashing rather than keeping the keys themselves makes the memory used while deduplicating proportional to the
    number of rows instead of to the size of the corpus, which matters for the larger retrieval datasets. Each key
    is length-prefixed so that a row cannot collide with a differently split one, e.g. `("a", "b")` and `("ab", "")`.
    """
    digest = hashlib.blake2b(digest_size=16)
    for key in keys:
        encoded = key.encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.digest()


def text_length(text: str, unit: TextLengthUnit) -> int:
    """The length of `text` in `unit`, ignoring surrounding whitespace."""
    stripped = _normalize(text)
    if unit == "characters":
        return len(stripped)
    return len(stripped.split())


def keep_first_occurrence(keys: Iterable[tuple[str, ...]]) -> list[int]:
    """Keep the rows whose comparison key has not been seen before.

    Args:
        keys: The comparison key of each row, one tuple per row with one entry per compared column.

    Returns:
        The indices of the first occurrence of each distinct row.
    """
    seen: set[bytes] = set()
    keep = []
    for i, row in enumerate(keys):
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        keep.append(i)
    return keep


def keep_long_enough(min_length: int, unit: TextLengthUnit) -> KeepIndicesFn:
    """Build a filter keeping the rows where *every* text is at least `min_length` `unit` long.

    Args:
        min_length: The minimum length a text must have to be kept.
        unit: Whether `min_length` counts characters or whitespace-separated words.

    Returns:
        A `KeepIndicesFn` applying the length threshold.
    """

    def _keep(keys: Iterable[tuple[str, ...]]) -> list[int]:
        return [
            i
            for i, row in enumerate(keys)
            if all(text_length(text, unit) >= min_length for text in row)
        ]

    return _keep


def iter_row_keys(
    dataset: Dataset,
    col_modalities: Mapping[str, Modalities],
    *,
    normalize: TextNormalization = "strip",
    num_proc: int | None = None,
) -> Iterator[tuple[str, ...]]:
    """Iterate the comparison key of each row, one entry per column of `col_modalities`.

    Text is compared by its normalized value; every other modality is compared by a hash of its content, using the
    same hash functions as the descriptive statistics. Text columns stay lazy so that the memory needed to compare
    a large text corpus does not grow with its size; the other modalities have to be decoded to be hashed.
    """
    per_column: list[Iterable[str]] = []
    for column, modality in col_modalities.items():
        if modality == "text":
            per_column.append(_normalize(value, normalize) for value in dataset[column])
        else:
            per_column.append(_HASH_FN[modality](dataset[column], max_workers=num_proc))
    return zip(*per_column)


def iter_texts(dataset: Dataset, columns: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Iterate the raw text of `columns` row by row, without materializing it all at once."""
    return zip(*(dataset[column] for column in columns))


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


def apply_row_filter(
    dataset: Dataset,
    col_modalities: Mapping[str, Modalities],
    keep_fn: KeepIndicesFn,
    *,
    normalize: TextNormalization = "strip",
    num_proc: int | None = None,
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
        keys = iter_row_keys(
            dataset, col_modalities, normalize=normalize, num_proc=num_proc
        )
        filtered = dataset.select(keep_fn(keys))

    return filtered, before - _count_values(filtered, columns[0], grouped)
