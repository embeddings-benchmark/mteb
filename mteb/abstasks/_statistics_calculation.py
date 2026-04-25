from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any, cast

from mteb.types.statistics import (
    AudioStatistics,
    ImageStatistics,
    LabelStatistics,
    PairModalityStatistics,
    RelevantDocsStatistics,
    ScoreStatistics,
    SingleInputModalityStatistics,
    TextStatistics,
    TopRankedStatistics,
    VideoStatistics,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from PIL import Image
    from torchcodec.decoders import VideoDecoder  # type: ignore[import-untyped]

    from mteb.types import Modalities, TopRankedDocumentsType
    from mteb.types._encoder_io import AudioInputItem


def compute_text_hashes(texts: list[str]) -> list[str]:
    """Return a hash per text — for text, the string itself is the identity key."""
    return texts


def compute_image_hashes(images: list[Image.Image]) -> list[str]:
    """Return a per-image MD5 hash of the raw pixel bytes."""
    return [
        hashlib.md5(img.tobytes(), usedforsecurity=False).hexdigest() for img in images
    ]


def compute_audio_hashes(audios: list[AudioInputItem]) -> list[str]:
    """Return a per-audio MD5 hash of the raw sample array bytes."""
    return [
        hashlib.md5(audio["array"].tobytes(), usedforsecurity=False).hexdigest()
        for audio in audios
    ]


def compute_video_hashes(videos: list[VideoDecoder]) -> list[str]:
    """Return a per-video MD5 hash derived from the first decoded frame.

    Decoding a frame is the most expensive part of video statistics; this function
    is extracted so callers can pass the resulting list to ``calculate_video_statistics``
    and avoid repeating the decode.
    """
    hashes = []
    for video in videos:
        meta = video.metadata
        num_frames = meta.num_frames
        avg_fps = meta.average_fps

        if num_frames is not None and avg_fps is not None and avg_fps > 0:
            # Sample one frame per second: indices 0, fps, 2*fps, ...
            step = max(1, round(avg_fps))
            frame_indices = list(range(0, num_frames, step))
        else:
            frame_indices = [0]

        frames = video.get_frames_at(frame_indices).data
        hashes.append(
            hashlib.md5(frames.numpy().tobytes(), usedforsecurity=False).hexdigest()
        )
    return hashes


def calculate_text_statistics(
    texts: list[str],
    hashes: list[str] | None = None,
) -> TextStatistics:
    """Calculate descriptive statistics for a list of texts.

    Args:
        texts: List of texts to analyze.
        hashes: Optional pre-computed identity keys (from :func:`compute_text_hashes`).
            When provided the function skips recomputing them.

    Returns:
        TextStatistics: A dictionary containing the descriptive statistics.
    """
    if hashes is None:
        hashes = compute_text_hashes(texts)
    lengths = [len(text) for text in texts]
    return TextStatistics(
        total_text_length=sum(lengths),
        min_text_length=min(lengths),
        average_text_length=sum(lengths) / len(lengths),
        max_text_length=max(lengths),
        unique_texts=len(set(hashes)),
    )


def calculate_image_statistics(
    images: list[Image.Image],
    hashes: list[str] | None = None,
) -> ImageStatistics:
    """Calculate descriptive statistics for a list of images.

    Args:
        images: List of images to analyze. Each image should have a ``size``
            attribute returning ``(width, height)``.
        hashes: Optional pre-computed MD5 hashes (from :func:`compute_image_hashes`).
            When provided the function skips recomputing them.

    Returns:
        ImageStatistics: A dictionary containing the descriptive statistics.
    """
    if hashes is None:
        hashes = compute_image_hashes(images)
    img_widths, img_heights = [], []
    for img in images:
        width, height = img.size
        img_heights.append(height)
        img_widths.append(width)

    return ImageStatistics(
        min_image_width=min(img_widths),
        average_image_width=sum(img_widths) / len(img_widths),
        max_image_width=max(img_widths),
        min_image_height=min(img_heights),
        average_image_height=sum(img_heights) / len(img_heights),
        max_image_height=max(img_heights),
        unique_images=len(set(hashes)),
    )


def calculate_audio_statistics(
    audios: list[AudioInputItem],
    hashes: list[str] | None = None,
) -> AudioStatistics:
    """Calculate descriptive statistics for a list of audio clips.

    Args:
        audios: List of audio clips to analyze. Each clip must have ``array``
            and ``sampling_rate`` keys.
        hashes: Optional pre-computed MD5 hashes (from :func:`compute_audio_hashes`).
            When provided the function skips recomputing them.

    Returns:
        A dictionary containing the descriptive statistics.
    """
    if hashes is None:
        hashes = compute_audio_hashes(audios)
    audio_lengths = []
    sampling_rates: dict[int, int] = defaultdict(int)

    for audio in audios:
        array = audio["array"]
        sampling_rate = audio["sampling_rate"]
        audio_lengths.append(len(array) / sampling_rate)
        sampling_rates[sampling_rate] += 1

    return AudioStatistics(
        total_duration_seconds=sum(audio_lengths),
        min_duration_seconds=min(audio_lengths),
        average_duration_seconds=sum(audio_lengths) / len(audio_lengths),
        max_duration_seconds=max(audio_lengths),
        unique_audios=len(set(hashes)),
        average_sampling_rate=(
            sum(rate * count for rate, count in sampling_rates.items()) / len(audios)
        ),
        sampling_rates=dict(sampling_rates),
    )


def calculate_video_statistics(  # noqa: PLR0914
    videos: list[VideoDecoder],
    hashes: list[str] | None = None,
) -> VideoStatistics:
    """Calculate descriptive statistics for a list of video clips.

    Args:
        videos: List of VideoDecoder objects to analyze.
        hashes: Optional pre-computed MD5 hashes (from :func:`compute_video_hashes`).
            When provided the function skips decoding the first frame again, which
            is the most expensive part of this function.

    Returns:
        A dictionary containing the descriptive statistics.
    """
    if hashes is None:
        hashes = compute_video_hashes(videos)
    durations: list[float | None] = []
    frames_counts: list[int | None] = []
    widths: list[int | None] = []
    heights: list[int | None] = []
    fps_counts: dict[int, int] = defaultdict(int)
    resolution_counts: dict[str, int] = defaultdict(int)

    for video in videos:
        meta = video.metadata

        num_frames = meta.num_frames
        avg_fps = meta.average_fps
        duration = meta.duration_seconds
        if (
            duration is None
            and num_frames is not None
            and avg_fps is not None
            and avg_fps > 0
        ):
            duration = num_frames / avg_fps

        durations.append(duration)
        frames_counts.append(num_frames)
        widths.append(meta.width)
        heights.append(meta.height)
        if avg_fps is not None:
            fps_counts[round(avg_fps)] += 1
        if meta.width is not None and meta.height is not None:
            resolution_counts[f"{meta.width}x{meta.height}"] += 1

    n = len(videos)
    all_durations = durations if None not in durations else None
    all_frames = frames_counts if None not in frames_counts else None
    all_widths = widths if None not in widths else None
    all_heights = heights if None not in heights else None
    has_all_fps = len(fps_counts) > 0 and sum(fps_counts.values()) == n
    has_all_resolutions = sum(resolution_counts.values()) == n

    if has_all_resolutions:
        resolutions_wh = [
            (int(r.split("x")[0]), int(r.split("x")[1])) for r in resolution_counts
        ]
        min_resolution: tuple[int, int] | None = min(
            resolutions_wh, key=lambda r: r[0] * r[1]
        )
        max_resolution: tuple[int, int] | None = max(
            resolutions_wh, key=lambda r: r[0] * r[1]
        )
    else:
        min_resolution = None
        max_resolution = None

    return VideoStatistics(
        total_duration_seconds=sum(all_durations)  # type: ignore[arg-type]
        if all_durations is not None
        else None,
        total_frames=sum(all_frames) if all_frames is not None else None,  # type: ignore[type-var,arg-type]
        min_width=min(all_widths) if all_widths is not None else None,  # type: ignore[type-var]
        average_width=sum(all_widths) / n if all_widths is not None else None,  # type: ignore[type-var,arg-type]
        max_width=max(all_widths) if all_widths is not None else None,  # type: ignore[type-var]
        min_height=min(all_heights) if all_heights is not None else None,  # type: ignore[type-var]
        average_height=sum(all_heights) / n if all_heights is not None else None,  # type: ignore[type-var,arg-type]
        max_height=max(all_heights) if all_heights is not None else None,  # type: ignore[type-var]
        min_duration_seconds=min(all_durations) if all_durations is not None else None,  # type: ignore[type-var]
        average_duration_seconds=sum(all_durations) / n  # type: ignore[type-var,arg-type]
        if all_durations is not None
        else None,
        max_duration_seconds=max(all_durations) if all_durations is not None else None,  # type: ignore[type-var]
        unique_videos=len(set(hashes)),
        average_fps=sum(rate * count for rate, count in fps_counts.items()) / n
        if has_all_fps
        else None,
        fps=dict(fps_counts),
        min_resolution=min_resolution,
        average_resolution=(sum(all_widths) / n, sum(all_heights) / n)  # type: ignore[arg-type]
        if all_widths is not None and all_heights is not None
        else None,
        max_resolution=max_resolution,
        resolutions=dict(resolution_counts),
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


def calculate_single_input_modality_statistics(
    col_inputs: dict[Modalities, list[Any]],
    hashes: dict[str, list[str]] | None = None,
) -> SingleInputModalityStatistics:
    """Compute per-modality statistics for a single-input dataset."""
    _hashes = hashes or {}
    return SingleInputModalityStatistics(
        text_statistics=calculate_text_statistics(
            col_inputs["text"], hashes=_hashes.get("text")
        )
        if "text" in col_inputs
        else None,
        image_statistics=calculate_image_statistics(
            col_inputs["image"], hashes=_hashes.get("image")
        )
        if "image" in col_inputs
        else None,
        audio_statistics=calculate_audio_statistics(
            col_inputs["audio"], hashes=_hashes.get("audio")
        )
        if "audio" in col_inputs
        else None,
        video_statistics=calculate_video_statistics(
            col_inputs["video"], hashes=_hashes.get("video")
        )
        if "video" in col_inputs
        else None,
    )


_STAT_FN: dict[str, Any] = {
    "text": calculate_text_statistics,
    "image": calculate_image_statistics,
    "audio": calculate_audio_statistics,
    "video": calculate_video_statistics,
}


def _compute_side_statistics(
    col_modalities: list[tuple[str, str]],
    load_col: Callable[[str], list],
    n: int,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Compute per-modality statistics and per-row hashes for one side of a pair.

    Args:
        col_modalities: List of ``(column_name, modality)`` pairs describing
            which dataset column carries which modality for this side.
        load_col: Callable that loads a column by name from the dataset.
        n: Number of samples.

    Returns a dict mapping ``{modality}_statistics`` keys to computed statistics
    objects (``None`` for absent modalities), and a list of per-row hash lists
    for use in unique-pair counting.
    """
    stats: dict[str, Any] = {
        "text_statistics": None,
        "image_statistics": None,
        "audio_statistics": None,
        "video_statistics": None,
    }
    all_hashes: list[list[str]] = [[] for _ in range(n)]
    for col, modality in col_modalities:
        data = load_col(col)
        hashes = _HASH_FN[modality](data)
        stats[f"{modality}_statistics"] = _STAT_FN[modality](data, hashes=hashes)
        for i, h in enumerate(hashes):
            all_hashes[i].append(h)
    return stats, all_hashes


def calculate_pair_modality_statistics(
    col_modalities1: list[tuple[str, str]],
    col_modalities2: list[tuple[str, str]],
    load_col: Callable[[str], list],
    n: int,
) -> PairModalityStatistics:
    """Compute per-modality statistics for a paired dataset.

    This is shared between STS and PairClassification tasks.

    Args:
        col_modalities1: ``[(column_name, modality), ...]`` for side 1.
        col_modalities2: ``[(column_name, modality), ...]`` for side 2.
        load_col: Callable that loads a column by name from the dataset split.
        n: Number of samples.
    """
    s1, all_h1 = _compute_side_statistics(col_modalities1, load_col, n)
    s2, all_h2 = _compute_side_statistics(col_modalities2, load_col, n)
    unique_pairs = len({(tuple(r1), tuple(r2)) for r1, r2 in zip(all_h1, all_h2)})

    return PairModalityStatistics(
        text1_statistics=s1["text_statistics"],
        text2_statistics=s2["text_statistics"],
        image1_statistics=s1["image_statistics"],
        image2_statistics=s2["image_statistics"],
        audio1_statistics=s1["audio_statistics"],
        audio2_statistics=s2["audio_statistics"],
        video1_statistics=s1["video_statistics"],
        video2_statistics=s2["video_statistics"],
        unique_pairs=unique_pairs,
    )


_HASH_FN: dict[str, Any] = {
    "text": compute_text_hashes,
    "image": compute_image_hashes,
    "audio": compute_audio_hashes,
    "video": compute_video_hashes,
}


def _compute_modality_hashes(
    col_inputs: dict[Modalities, list[Any]],
) -> dict[str, list[str]]:
    """Compute per-sample hashes for each modality using the shared hash functions.

    Reuses the same hashing logic as the ``calculate_*_statistics`` functions so that
    callers can pass the result to both statistics functions and intersection checks
    without decoding the data twice.
    """
    return {mod: _HASH_FN[mod](values) for mod, values in col_inputs.items()}


def _count_samples_in_train(
    test_hashes: dict[str, list[str]],
    train_hashes: dict[str, list[str]] | None,
) -> int | None:
    """Count unique test samples (by row-tuple hash) that also appear in the train split.

    Args:
        test_hashes: Per-modality hash lists for the test split (from :func:`_compute_modality_hashes`).
        train_hashes: Per-modality hash lists for the train split, or ``None`` when
            the evaluated split *is* the train split.

    Returns:
        Number of unique test row-tuples present in train, or ``None``.
    """
    if train_hashes is None:
        return None
    mods = sorted(test_hashes.keys())
    n_test = len(test_hashes[mods[0]])
    n_train = len(train_hashes[mods[0]])
    test_keys = {tuple(test_hashes[m][i] for m in mods) for i in range(n_test)}
    train_keys = {tuple(train_hashes[m][i] for m in mods) for i in range(n_train)}
    return len(test_keys & train_keys)
