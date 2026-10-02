"""Removing samples whose content is too small to carry meaning: a short text, a small image, a brief clip."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from mteb.abstasks._statistics_calculation import (
    _audio_duration_seconds,
    _video_duration_seconds,
)

from ._filtering import _CleaningFilter, _filter_task_rows, _keep_at_least

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from PIL import Image

    from mteb.types import HFSubset, Modalities

    from ._filtering import T


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
        _CleaningFilter(
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
        min_length: The shortest length a text may have. `1` removes only empty and whitespace-only texts. What
            counts as too short depends on the language, as a character carries more meaning in e.g. Chinese than
            in English.
        length_fn: How to measure a text. Defaults to its number of characters. Counting words with
            `lambda text: len(text.split())` only holds for scripts that separate them by spaces.
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
