from __future__ import annotations

from collections import Counter
from typing import Any

from mteb.types.statistics import (
    ImageStatistics,
    LabelStatistics,
    ScoreStatistics,
    TextStatistics,
)


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


def calculate_image_statistics(images: list[Any]) -> ImageStatistics:
    img_widths, img_heights = [], []
    for img in images:
        width, height = img.size  # type: ignore
        img_heights.append(height)
        img_widths.append(width)

    return ImageStatistics(
        min_image_width=min(img_widths),
        average_image_width=sum(img_widths) / len(img_widths),
        max_image_width=max(img_widths),
        min_image_height=min(img_heights),
        average_image_height=sum(img_heights) / len(img_heights),
        max_image_height=max(img_heights),
    )


def calculate_label_statistics(labels: list[int | list[int]]) -> LabelStatistics:
    if isinstance(labels[0], int):
        label_len = [1] * len(labels)
        total_label_len = len(labels)
        total_labels = labels
    elif isinstance(labels[0], list):
        # multilabel classification
        label_len = [len(l) for l in labels]
        total_label_len = sum(label_len)
        total_labels = []
        for l in labels:
            total_labels.extend(l if len(l) > 0 else [None])
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
