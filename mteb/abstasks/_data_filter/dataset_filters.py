"""Column-aware row filters backing the public `AbsTask` data cleaning API.

The functions here operate on a single `datasets.Dataset` and are agnostic to the task type: the caller supplies the
names of the text columns to look at, and a `KeepIndicesFn` deciding which of the texts to keep.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from datasets import Dataset

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
"""

_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)

KeepIndicesFn = Callable[[Iterable[tuple[str, ...]]], list[int]]
"""Given the text of each row, return the (ascending) indices of the rows to keep.

The texts are passed as a lazy iterable and may only be consumed once, so that filtering a large corpus does not
require holding all of its text in memory at the same time.
"""


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


def text_key(texts: tuple[str, ...], normalize: TextNormalization = "strip") -> bytes:
    """A compact key identifying a row by its normalized text.

    Hashing rather than keeping the text itself makes the memory used while deduplicating proportional to the number
    of rows instead of to the size of the corpus, which matters for the larger retrieval datasets. Each text is
    length-prefixed so that a row cannot collide with a differently split one, e.g. `("a", "b")` and `("ab", "")`.
    """
    digest = hashlib.blake2b(digest_size=16)
    for text in texts:
        encoded = _normalize(text, normalize).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.digest()


def text_length(text: str, unit: TextLengthUnit) -> int:
    """The length of `text` in `unit`, ignoring surrounding whitespace."""
    stripped = _normalize(text)
    if unit == "characters":
        return len(stripped)
    return len(stripped.split())


def keep_first_occurrence(normalize: TextNormalization) -> KeepIndicesFn:
    """Build a filter keeping the rows whose texts have not been seen before.

    Args:
        normalize: How much of a difference between two texts to ignore when comparing them.

    Returns:
        A `KeepIndicesFn` returning the indices of the first occurrence of each distinct row.
    """

    def _keep(texts: Iterable[tuple[str, ...]]) -> list[int]:
        seen: set[bytes] = set()
        keep = []
        for i, row in enumerate(texts):
            key = text_key(row, normalize)
            if key in seen:
                continue
            seen.add(key)
            keep.append(i)
        return keep

    return _keep


def keep_long_enough(min_length: int, unit: TextLengthUnit) -> KeepIndicesFn:
    """Build a filter keeping the rows where *every* text is at least `min_length` `unit` long.

    Args:
        min_length: The minimum length a text must have to be kept.
        unit: Whether `min_length` counts characters or whitespace-separated words.

    Returns:
        A `KeepIndicesFn` applying the length threshold.
    """

    def _keep(texts: Iterable[tuple[str, ...]]) -> list[int]:
        return [
            i
            for i, row in enumerate(texts)
            if all(text_length(text, unit) >= min_length for text in row)
        ]

    return _keep


def iter_texts(dataset: Dataset, columns: Sequence[str]) -> Iterator[tuple[str, ...]]:
    """Iterate the text of `columns` row by row, without materializing them all at once."""
    return zip(*(dataset[column] for column in columns))


def _is_grouped(dataset: Dataset, columns: Sequence[str]) -> bool:
    """Whether each row holds a list of texts (e.g. the sentences of a cluster) rather than a single text."""
    return isinstance(dataset[0][columns[0]], list)


def _filter_within_row(
    example: dict[str, Any], columns: Sequence[str], keep_fn: KeepIndicesFn
) -> dict[str, Any]:
    """Apply `keep_fn` inside a single row of a grouped dataset.

    Every other column of the row that is a list of the same length is filtered alongside the text columns, which
    keeps parallel columns such as the cluster labels aligned with their texts.
    """
    lengths = {len(example[column]) for column in columns}
    if len(lengths) != 1:
        raise ValueError(
            f"The grouped columns {list(columns)} of a row have differing lengths {sorted(lengths)}, "
            "so they cannot be filtered together."
        )
    n_texts = lengths.pop()
    keep = keep_fn(
        tuple(example[column][i] for column in columns) for i in range(n_texts)
    )
    return {
        column: [value[i] for i in keep]
        if isinstance(value, list) and len(value) == n_texts
        else value
        for column, value in example.items()
    }


def _count_texts(dataset: Dataset, column: str, grouped: bool) -> int:
    if not grouped:
        return len(dataset)
    return sum(len(texts) for texts in dataset[column])


def apply_row_filter(
    dataset: Dataset,
    columns: Sequence[str],
    keep_fn: KeepIndicesFn,
    *,
    num_proc: int | None = None,
) -> tuple[Dataset, int]:
    """Filter `dataset` down to the texts that `keep_fn` keeps.

    For a regular dataset this drops whole rows. For a grouped dataset -- one where each row holds a list of texts,
    as clustering tasks do -- the filter is applied within each row instead, and the parallel columns of that row
    (e.g. the labels) are filtered along with it.

    Args:
        dataset: The dataset to filter.
        columns: The text columns to hand to `keep_fn`.
        keep_fn: Decides which texts to keep.
        num_proc: Number of processes to use when filtering a grouped dataset.

    Returns:
        The filtered dataset and the number of texts that were removed.

    Raises:
        ValueError: If one of `columns` is not in the dataset.
    """
    missing = [column for column in columns if column not in dataset.column_names]
    if missing:
        raise ValueError(
            f"Cannot filter on {missing}: the dataset only has the columns {dataset.column_names}."
        )
    if len(dataset) == 0:
        return dataset, 0

    grouped = _is_grouped(dataset, columns)
    before = _count_texts(dataset, columns[0], grouped)

    if grouped:
        filtered = dataset.map(
            _filter_within_row,
            fn_kwargs={"columns": columns, "keep_fn": keep_fn},
            num_proc=num_proc,
        )
    else:
        filtered = dataset.select(keep_fn(iter_texts(dataset, columns)))

    return filtered, before - _count_texts(filtered, columns[0], grouped)
