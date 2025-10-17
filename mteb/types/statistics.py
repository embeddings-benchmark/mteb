from typing_extensions import NotRequired, TypedDict

from mteb.types import HFSubset


class SplitDescriptiveStatistics(TypedDict):
    """Base class for descriptive statistics for the subset."""

    pass


class DescriptiveStatistics(TypedDict, SplitDescriptiveStatistics):
    """Class for descriptive statistics for the full task."""

    hf_subset_descriptive_stats: NotRequired[dict[HFSubset, SplitDescriptiveStatistics]]


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

        unique_images: Number of unique images
    """

    min_image_width: float
    average_image_width: float
    max_image_width: float

    min_image_height: float
    average_image_height: float
    max_image_height: float

    unique_images: int


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


class TopRankedStatistics(TypedDict):
    """Statistics for top ranked documents in a retrieval task.

    Attributes:
        num_top_ranked: Total number of top ranked documents across all queries.
        min_top_ranked_per_query: Minimum number of top ranked documents for any query.
        average_top_ranked_per_query: Average number of top ranked documents per query.
        max_top_ranked_per_query: Maximum number of top ranked documents for any query.
    """

    num_top_ranked: int
    min_top_ranked_per_query: int
    average_top_ranked_per_query: float
    max_top_ranked_per_query: int


class RelevantDocsStatistics(TypedDict):
    """Statistics for relevant documents in a retrieval task.

    Attributes:
        num_relevant_docs: Total number of relevant documents across all queries.
        min_relevant_docs_per_query: Minimum number of relevant documents for any query.
        average_relevant_docs_per_query: Average number of relevant documents per query.
        max_relevant_docs_per_query: Maximum number of relevant documents for any query.
        unique_relevant_docs: Number of unique relevant documents across all queries.
    """

    num_relevant_docs: int
    min_relevant_docs_per_query: int
    average_relevant_docs_per_query: float
    max_relevant_docs_per_query: float
    unique_relevant_docs: int
