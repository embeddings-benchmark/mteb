"""Removing samples whose content is too small or too large: a text's length, an image's size, a clip's duration."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from mteb.abstasks._statistics_calculation import (
    _audio_duration_seconds,
    _video_duration_seconds,
)

from ._filtering import _CleaningFilter, _filter_task_rows, _keep_within_bounds

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from PIL import Image

    from mteb.types import HFSubset, Modalities

    from ._filtering import T


def _remove_outside_bounds(
    task: T,
    name: str,
    modality: Modalities,
    measure_fn: Callable[[Any], float | None],
    *,
    bounds: Mapping[str, float | None],
    columns: Sequence[str] | None,
    splits: Sequence[str] | None,
    subsets: Sequence[HFSubset] | None,
    num_proc: int | None,
) -> T:
    """Remove the samples holding a `modality` value that `measure_fn` finds outside the bounds.

    Args:
        task: The task to filter.
        name: The filter's name, as the caller exposes it.
        modality: The modality to measure.
        measure_fn: The size of a value, or None if it cannot be told.
        bounds: The lower and then the upper bound, each under the name the caller takes it as, so that a message
            about them can name the argument the caller passed.
        columns: The columns to measure. Defaults to every column of `modality`.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use.

    Returns:
        A copy of the task holding the filtered data.

    Raises:
        ValueError: If neither bound is given, or if they cannot be met together.
    """
    (lower, minimum), (upper, maximum) = bounds.items()
    if minimum is None and maximum is None:
        raise ValueError(f"`{name}` needs a bound: pass `{lower}`, `{upper}`, or both.")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ValueError(
            f"`{name}` was given {lower}={minimum} above {upper}={maximum}, which no sample can meet."
        )

    return _filter_task_rows(
        task,
        _CleaningFilter(
            name,
            functools.partial(
                _keep_within_bounds,
                minimum=minimum,
                maximum=maximum,
                measure_fn=measure_fn,
            ),
            modalities=frozenset({modality}),
            compares_rows=False,
        ),
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def remove_by_text_length(
    task: T,
    *,
    min_length: int | None = None,
    max_length: int | None = None,
    length_fn: Callable[[str], int] = len,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove samples with a text shorter than `min_length` or longer than `max_length`.

    A sample is removed when any of its texts is out of bounds, e.g. either sentence of a pair. Texts are counted in
    characters, ignoring surrounding whitespace, so `min_length=1` removes the empty and whitespace-only ones; a
    missing text counts as empty. A retrieval document is measured on its title and text together and a query on its
    text and instruction, as they are encoded that way when evaluated: a removed document takes its relevance
    judgements with it, and a query left without one is dropped.

    Only text is measured, so images, audio and video are kept whatever their size, as is a retrieval document or
    query that combines text with one of them, e.g. a page whose image carries the content its text lacks.

    Args:
        task: The task to filter. It is not modified.
        min_length: The shortest length a text may have. `1` removes only empty and whitespace-only texts. What
            counts as too short depends on the language, as a character carries more meaning in e.g. Chinese than
            in English.
        max_length: The longest length a text may have, for dropping the outliers a corpus sometimes holds, such as
            the 1M character document of MSMARCOv2 against its 341 character average.
        length_fn: How to measure a text. Defaults to its number of characters. Counting words with
            `lambda text: len(text.split())` only holds for scripts that separate them by spaces.
        columns: The text columns to measure. Defaults to every text column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading and filtering the dataset.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `MassiveIntentClassification (remove_by_text_length)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If neither bound is given, if they cannot be met together, if `task` has no text, or if `splits`
            and `subsets` together match none of its splits.
        KeyError: If `columns` names a column that is not a text column of the task.

    Examples:
        >>> import mteb
        >>> from mteb.data_cleaning import remove_by_text_length
        >>> task = mteb.get_task("MassiveIntentClassification")
        >>> cleaned = remove_by_text_length(task, min_length=1)  # empty and whitespace-only texts
        >>> cleaned = remove_by_text_length(task, min_length=3, max_length=10_000)
        >>> cleaned = remove_by_text_length(task, min_length=3, length_fn=lambda text: len(text.split()))
    """
    return _remove_outside_bounds(
        task,
        "remove_by_text_length",
        "text",
        length_fn,
        bounds={"min_length": min_length, "max_length": max_length},
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def _shorter_side(image: Image.Image) -> int:
    """The shorter of an image's width and height, so that a thin image counts as small. The default."""
    return min(image.size)


def remove_by_image_size(
    task: T,
    *,
    min_size: int | None = None,
    max_size: int | None = None,
    size_fn: Callable[[Image.Image], float] = _shorter_side,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove samples with an image smaller than `min_size` or larger than `max_size`, e.g. a 1x1 placeholder.

    A sample is removed when any of its images is out of bounds. An image is as small as its shorter side by default,
    so the bounds are the width and the height it must stay within, as the `min_image_width` and `min_image_height` of
    the task's descriptive statistics report them. Only images are measured, so the texts of a sample are kept
    whatever their length.

    Args:
        task: The task to filter. It is not modified.
        min_size: The smallest size, by `size_fn`, an image may have.
        max_size: The largest size, by `size_fn`, an image may have.
        size_fn: How to measure an image. Defaults to its shorter side in pixels; `lambda image: image.width *
            image.height` measures its area instead, in which case the bounds are numbers of pixels.
        columns: The image columns to measure. Defaults to every image column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `ROxfordEasyI2IRetrieval (remove_by_image_size)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If neither bound is given, if they cannot be met together, if `task` has no images, or if
            `splits` and `subsets` together match none of its splits.
        KeyError: If `columns` names a column that is not an image column of the task.
    """
    return _remove_outside_bounds(
        task,
        "remove_by_image_size",
        "image",
        size_fn,
        bounds={"min_size": min_size, "max_size": max_size},
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def remove_by_audio_duration(
    task: T,
    *,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove samples with an audio clip shorter than `min_seconds` or longer than `max_seconds`.

    A sample is removed when any of its clips is out of bounds, measured as the `min_duration_seconds` of the task's
    descriptive statistics measures it. Only audio is measured, so the texts of a sample are kept whatever their
    length.

    Args:
        task: The task to filter. It is not modified.
        min_seconds: The shortest duration, in seconds, an audio clip may have.
        max_seconds: The longest duration, in seconds, an audio clip may have.
        columns: The audio columns to measure. Defaults to every audio column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `BeijingOpera (remove_by_audio_duration)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If neither bound is given, if they cannot be met together, if `task` has no audio, or if `splits`
            and `subsets` together match none of its splits.
        KeyError: If `columns` names a column that is not an audio column of the task.
    """
    return _remove_outside_bounds(
        task,
        "remove_by_audio_duration",
        "audio",
        _audio_duration_seconds,
        bounds={"min_seconds": min_seconds, "max_seconds": max_seconds},
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )


def remove_by_video_duration(
    task: T,
    *,
    min_seconds: float | None = None,
    max_seconds: float | None = None,
    columns: Sequence[str] | None = None,
    splits: Sequence[str] | None = None,
    subsets: Sequence[HFSubset] | None = None,
    num_proc: int | None = None,
) -> T:
    """Remove samples with a video outside the bounds. A video whose duration is unknown is kept.

    A sample is removed when any of its videos is out of bounds, measured as the `min_duration_seconds` of the task's
    descriptive statistics measures it. Only video is measured, so the texts of a sample are kept whatever their
    length.

    Args:
        task: The task to filter. It is not modified.
        min_seconds: The shortest duration, in seconds, a video may have.
        max_seconds: The longest duration, in seconds, a video may have.
        columns: The video columns to measure. Defaults to every video column of the task.
        splits: The splits to filter. Defaults to every split of the dataset.
        subsets: The Huggingface subsets to filter. Defaults to every loaded subset.
        num_proc: Number of processes to use for loading the dataset.

    Returns:
        A copy of the task holding the filtered data, named after the filters applied to it, e.g.
        `CoVRRVT2VRetrieval (remove_by_video_duration)`. The task passed in is left untouched.

    Raises:
        NotImplementedError: If `task` aggregates other tasks, which hold the data instead.
        ValueError: If neither bound is given, if they cannot be met together, if `task` has no videos, or if
            `splits` and `subsets` together match none of its splits.
        KeyError: If `columns` names a column that is not a video column of the task.
    """
    return _remove_outside_bounds(
        task,
        "remove_by_video_duration",
        "video",
        _video_duration_seconds,
        bounds={"min_seconds": min_seconds, "max_seconds": max_seconds},
        columns=columns,
        splits=splits,
        subsets=subsets,
        num_proc=num_proc,
    )
