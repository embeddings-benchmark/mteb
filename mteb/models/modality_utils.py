"""Helpers for models that encode datasets with interleaved modalities.

A dataset may interleave samples with different modality coverage — some rows
carrying only text, others only an image, others both:

```python
{"text": ["text 1", "", "text 3"], "image": [None, image1, None]}
```

A batch therefore always has a column per declared modality, but an individual row
may not carry a value for it. `""` marks an absent text (the dataloader normalizes
`None` to it, so that text encoders never see a `None`) and `None` marks any other
absent modality.

Models that fuse modalities per row should encode only the rows that carry each
modality and combine the results by row, rather than assuming every row carries
every modality.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mteb.types import BatchedInput, Modalities


def is_modality_present(value: object) -> bool:
    """Whether a single value in a batch column carries data.

    Args:
        value: One entry of a modality column, e.g. `batch["image"][3]`.

    Returns:
        False if the row carries nothing for that modality — `None`, the empty
        string used for absent text, or an empty list of images/frames — else True.
    """
    if value is None:
        return False
    if isinstance(value, (str, list, tuple)):
        return len(value) > 0
    return True


def _modality_column(
    batch: BatchedInput, modality: Modalities
) -> Sequence[object] | None:
    """The batch column for `modality`, or None if the batch has no such column.

    `BatchedInput` is a union of TypedDicts, so it has to be widened to a plain
    mapping to be looked up by a modality name that is not a literal.
    """
    return cast("Mapping[str, Sequence[object]]", batch).get(modality)


def get_present_indices(batch: BatchedInput, modality: Modalities) -> list[int]:
    """Indices of the rows of `batch` that carry data for `modality`.

    Args:
        batch: A batch yielded by the dataloader.
        modality: The modality to look for, e.g. `"image"`.

    Returns:
        The row indices carrying that modality, empty if the batch has no such
        column at all.

    Example:
        ```python
        image_rows = get_present_indices(batch, "image")
        images = [batch["image"][i] for i in image_rows]
        ```
    """
    values = _modality_column(batch, modality)
    if values is None:
        return []
    return [i for i, value in enumerate(values) if is_modality_present(value)]


def is_interleaved(batch: BatchedInput, modality: Modalities) -> bool:
    """Whether `modality` is present on some but not all rows of `batch`.

    Args:
        batch: A batch yielded by the dataloader.
        modality: The modality to check.

    Returns:
        True when the batch mixes rows that carry the modality with rows that do
        not. A model that handles only uniform batches can use this to report an
        informative error instead of failing inside its processor.
    """
    values = _modality_column(batch, modality)
    if not values:
        return False
    return 0 < len(get_present_indices(batch, modality)) < len(values)
