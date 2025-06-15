from __future__ import annotations

from typing import TypedDict


class DescriptiveStatistics(TypedDict):
    """Class for descriptive statistics."""

    pass


class TextStatistics(TypedDict):
    """Class for descriptive statistics for texts.

    Attributes:
        total_text_length: Total length of all texts
        min_text_length: Minimum length of text
        average_text_length: Average length of text
        max_text_length: Maximum length of text
        unique_texts: Number of unique texts
    """

    total_text_length: int
    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_texts: int


class ImageStatistics(TypedDict):
    """Class for descriptive statistics for images.

    Attributes:
        min_image_width: Minimum width of images
        average_image_width: Average width of images
        max_image_width: Maximum width of images

        min_image_height: Minimum height of images
        average_image_height: Average height of images
        max_image_height: Maximum height of images
    """

    min_image_width: float
    average_image_width: float
    max_image_width: float

    min_image_height: float
    average_image_height: float
    max_image_height: float


class LabelStatistics(TypedDict):
    """Class for descriptive statistics for texts.

    Attributes:
        min_labels_per_text: Minimum number of labels per text
        average_label_per_text: Average number of labels per text
        max_labels_per_text: Maximum number of labels per text

        unique_labels: Number of unique labels
        labels: dict of label frequencies
    """

    min_labels_per_text: int
    average_label_per_text: float
    max_labels_per_text: int

    unique_labels: int
    labels: dict[str, dict[str, int]]


class ScoreStatistics(TypedDict):
    """Class for descriptive statistics for texts.

    Attributes:
        min_score: Minimum score
        avg_score: Average score
        max_score: Maximum score
    """

    min_score: int
    avg_score: float
    max_score: int
