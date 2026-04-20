from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, cast

from mteb.types.statistics import (
    AudioStatistics,
    ImageStatistics,
    LabelStatistics,
    RelevantDocsStatistics,
    ScoreStatistics,
    TextStatistics,
    TopRankedStatistics,
    VideoStatistics,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from PIL import Image
    from torchcodec.decoders import VideoDecoder  # type: ignore[import-untyped]

    from mteb.types import TopRankedDocumentsType
    from mteb.types._encoder_io import AudioInputItem


def calculate_text_statistics(texts: list[str]) -> TextStatistics:
    """Calculate descriptive statistics for a list of texts.

    Args:
        texts: List of texts to analyze.

    Returns:
        TextStatistics: A dictionary containing the descriptive statistics.
    """
    lengths = [len(text) for text in texts]
    unique_texts = len(set(texts))

    return TextStatistics(
        total_text_length=sum(lengths),
        min_text_length=min(lengths),
        average_text_length=sum(lengths) / len(lengths),
        max_text_length=max(lengths),
        unique_texts=unique_texts,
    )


def calculate_image_statistics(images: list[Image.Image]) -> ImageStatistics:
    """Calculate descriptive statistics for a list of images.

    Args:
        images: List of images to analyze. Each image should have a `size` attribute that returns a tuple (width, height).

    Returns:
        ImageStatistics: A dictionary containing the descriptive statistics.
    """
    img_widths, img_heights = [], []
    seen_hashes: set[str] = set()

    for img in images:
        width, height = img.size
        img_heights.append(height)
        img_widths.append(width)

        img_bytes = img.tobytes()
        img_hash = hashlib.md5(img_bytes, usedforsecurity=False).hexdigest()
        seen_hashes.add(img_hash)

    return ImageStatistics(
        min_image_width=min(img_widths),
        average_image_width=sum(img_widths) / len(img_widths),
        max_image_width=max(img_widths),
        min_image_height=min(img_heights),
        average_image_height=sum(img_heights) / len(img_heights),
        max_image_height=max(img_heights),
        # some image types (PngImageFile) may be unhashable
        unique_images=len(seen_hashes),
    )


def calculate_audio_statistics(audios: list[AudioInputItem]) -> AudioStatistics:
    """Calculate descriptive statistics for a list of audio clips.

    Args:
        audios: List of audio clips to analyze. Each audio clip should be a dictionary with 'array' and 'sampling_rate' keys.

    Returns:
        A dictionary containing the descriptive statistics.
    """
    audio_lengths = []
    sampling_rates: dict[int, int] = defaultdict(int)
    unique_audios = set()

    for audio in audios:
        array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        length_in_seconds = len(array) / sampling_rate
        audio_lengths.append(length_in_seconds)
        sampling_rates[sampling_rate] += 1

        audio_bytes = array.tobytes()
        audio_hash = hashlib.md5(audio_bytes, usedforsecurity=False).hexdigest()
        unique_audios.add(audio_hash)

    return AudioStatistics(
        total_duration_seconds=sum(audio_lengths),
        min_duration_seconds=min(audio_lengths),
        average_duration_seconds=sum(audio_lengths) / len(audio_lengths),
        max_duration_seconds=max(audio_lengths),
        unique_audios=len(unique_audios),
        average_sampling_rate=(
            sum(rate * count for rate, count in sampling_rates.items()) / len(audios)
        ),
        sampling_rates=dict(sampling_rates),
    )


def calculate_video_statistics(videos: list[VideoDecoder]) -> VideoStatistics:
    """Calculate descriptive statistics for a list of video clips.

    Args:
        videos: List of VideoDecoder objects to analyze.

    Returns:
        A dictionary containing the descriptive statistics.
    """
    durations = []
    frames_counts = []
    widths = []
    heights = []
    fps_counts: dict[int, int] = defaultdict(int)
    unique_videos: set[str] = set()

    for video in videos:
        meta = video.metadata

        num_frames = meta.num_frames or 0
        avg_fps = meta.average_fps or 0.0
        duration = meta.duration_seconds
        if duration is None:
            duration = num_frames / avg_fps if avg_fps > 0 else 0.0
        width = meta.width or 0
        height = meta.height or 0

        durations.append(duration)
        frames_counts.append(num_frames)
        widths.append(width)
        heights.append(height)
        fps_counts[round(avg_fps)] += 1

        first_frame = video.get_frames_at([0]).data
        video_hash = hashlib.md5(
            first_frame.numpy().tobytes(), usedforsecurity=False
        ).hexdigest()
        unique_videos.add(video_hash)

    n = len(videos)
    return VideoStatistics(
        total_duration_seconds=sum(durations),
        total_frames=sum(frames_counts),
        min_width=min(widths),
        average_width=sum(widths) / n,
        max_width=max(widths),
        min_height=min(heights),
        average_height=sum(heights) / n,
        max_height=max(heights),
        min_duration_seconds=min(durations),
        average_duration_seconds=sum(durations) / n,
        max_duration_seconds=max(durations),
        unique_videos=len(unique_videos),
        average_fps=sum(rate * count for rate, count in fps_counts.items()) / n,
        fps=dict(fps_counts),
    )


def calculate_label_statistics(labels: list[int | list[int]]) -> LabelStatistics:
    """Calculate descriptive statistics for a list of labels.

    Args:
        labels: List of labels, where each label can be an integer or a list of integers (for multilabel classification).

    Returns:
        LabelStatistics: A dictionary containing the descriptive statistics.

    """
    total_labels: list[int | None] = []

    if not isinstance(labels[0], list):
        # single label classification
        single_label = cast("list[int]", labels)
        label_len = [1] * len(single_label)
        total_label_len = len(single_label)
        total_labels.extend(single_label)
    elif isinstance(labels[0], list):
        # multilabel classification
        multilabel_labels = cast("list[list[int]]", labels)
        label_len = [len(l) for l in multilabel_labels]
        total_label_len = sum(label_len)
        for l in multilabel_labels:
            if l and len(l) > 0:
                total_labels.extend(l)
            else:
                total_labels.append(None)
    else:
        raise ValueError(
            "Labels must be a list of integers or a list of lists of integers."
        )

    label_count = Counter(total_labels)
    return LabelStatistics(
        min_labels_per_text=min(label_len),
        average_label_per_text=total_label_len / len(labels),
        max_labels_per_text=max(label_len),
        unique_labels=len(label_count),
        labels={
            str(label): {
                "count": value,
            }
            for label, value in label_count.items()
        },
    )


def calculate_score_statistics(scores: list[int | float]) -> ScoreStatistics:
    """Calculate descriptive statistics for a list of scores.

    Args:
        scores: List of scores to analyze.

    Returns:
        ScoreStatistics: A dictionary containing the descriptive statistics.
    """
    return ScoreStatistics(
        min_score=min(scores),
        avg_score=sum(scores) / len(scores),
        max_score=max(scores),
    )


def calculate_top_ranked_statistics(
    top_ranked: TopRankedDocumentsType, num_queries: int
) -> TopRankedStatistics:
    """Calculate statistics for top-ranked items.

    Args:
        top_ranked: List of lists, where each inner list contains IDs of top-ranked items.
        num_queries: Total number of queries.

    Returns:
        dict: A dictionary with the count of top-ranked items per ID.
    """
    return TopRankedStatistics(
        num_top_ranked=sum(
            len(docs) for docs in top_ranked.values() if docs is not None
        ),
        min_top_ranked_per_query=min(
            len(docs) for docs in top_ranked.values() if docs is not None
        ),
        average_top_ranked_per_query=(
            sum(len(docs) for docs in top_ranked.values() if docs is not None)
            / num_queries
        ),
        max_top_ranked_per_query=max(
            len(docs) for docs in top_ranked.values() if docs is not None
        ),
    )


def calculate_relevant_docs_statistics(
    relevant_docs: Mapping[str, Mapping[str, int]],
) -> RelevantDocsStatistics:
    qrels_lengths = [len(relevant_docs[qid]) for qid in relevant_docs]
    unique_qrels = len({doc for qid in relevant_docs for doc in relevant_docs[qid]})
    # number of qrels that are not 0
    num_qrels_non_zero = sum(
        sum(1 for doc_id in docs if docs[doc_id] != 0)
        for docs in relevant_docs.values()
    )
    qrels_per_doc = num_qrels_non_zero / len(relevant_docs)

    return RelevantDocsStatistics(
        num_relevant_docs=num_qrels_non_zero,
        min_relevant_docs_per_query=min(qrels_lengths),
        average_relevant_docs_per_query=qrels_per_doc,
        max_relevant_docs_per_query=max(qrels_lengths),
        unique_relevant_docs=unique_qrels,
    )
