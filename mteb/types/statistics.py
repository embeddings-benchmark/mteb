from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from typing_extensions import NotRequired

    from mteb.types import HFSubset


class SplitDescriptiveStatistics(TypedDict):
    """Base class for descriptive statistics for the subset."""

    pass


class DescriptiveStatistics(TypedDict, SplitDescriptiveStatistics):
    """Class for descriptive statistics for the full task.

    Attributes:
        num_samples: Total number of samples
        hf_subset_descriptive_stats: HFSubset descriptive statistics (only for multilingual datasets)
    """

    num_samples: int
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


class AudioStatistics(TypedDict):
    """Class for descriptive statistics for audio.

    Attributes:
        total_duration_seconds: Total length of all audio clips in total frames
        min_duration_seconds: Minimum length of audio clip in seconds
        average_duration_seconds: Average length of audio clip in seconds
        max_duration_seconds: Maximum length of audio clip in seconds
        unique_audios: Number of unique audio clips
        average_sampling_rate: Average sampling rate
        sampling_rates: Dict of unique sampling rates and their frequencies
    """

    total_duration_seconds: float

    min_duration_seconds: float
    average_duration_seconds: float
    max_duration_seconds: float

    unique_audios: int

    average_sampling_rate: float
    sampling_rates: dict[int, int]


class VideoStatistics(TypedDict):
    """Class for descriptive statistics for video.

    Attributes:
        total_duration_seconds: Total duration of all video clips in seconds
        total_frames: Total number of frames across all video clips

        min_width: Minimum width of video frames
        average_width: Average width of video frames
        max_width: Maximum width of video frames

        min_height: Minimum height of video frames
        average_height: Average height of video frames
        max_height: Maximum height of video frames

        min_duration_seconds: Minimum duration of a video clip in seconds
        average_duration_seconds: Average duration of a video clip in seconds
        max_duration_seconds: Maximum duration of a video clip in seconds

        unique_videos: Number of unique video clips

        average_fps: Average frames per second across all video clips
        fps: Dict of unique (rounded) fps values and their frequencies

        min_resolution: Resolution (width, height) with the smallest area
        average_resolution: Average resolution (average_width, average_height)
        max_resolution: Resolution (width, height) with the largest area
        resolutions: Dict mapping "WxH" resolution strings to their frequency counts
    """

    total_duration_seconds: float | None
    total_frames: int | None

    min_width: int | None
    average_width: float | None
    max_width: int | None

    min_height: int | None
    average_height: float | None
    max_height: int | None

    min_duration_seconds: float | None
    average_duration_seconds: float | None
    max_duration_seconds: float | None

    unique_videos: int

    average_fps: float | None
    fps: dict[int, int]

    min_resolution: tuple[int, int] | None
    average_resolution: tuple[float, float] | None
    max_resolution: tuple[int, int] | None
    resolutions: dict[str, int]


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

    min_score: int | float
    avg_score: float
    max_score: int | float


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


class SingleInputModalityStatistics(TypedDict):
    """Per-modality statistics for a single-input dataset (Classification, Regression, …).

    Fields are ``None`` when the corresponding modality is absent from the task.

    Attributes:
        text_statistics: Statistics for the text column.
        image_statistics: Statistics for the image column.
        audio_statistics: Statistics for the audio column.
        video_statistics: Statistics for the video column.
    """

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    audio_statistics: AudioStatistics | None
    video_statistics: VideoStatistics | None


class PairModalityStatistics(TypedDict):
    """Per-modality statistics for a paired dataset (STS, PairClassification, …).

    Each modality has a ``*1_statistics`` field for the first item in the pair
    and a ``*2_statistics`` field for the second item.  Fields are ``None`` when
    the corresponding modality is absent from the task.

    Attributes:
        text1_statistics: Text statistics for the first item.
        text2_statistics: Text statistics for the second item.
        image1_statistics: Image statistics for the first item.
        image2_statistics: Image statistics for the second item.
        audio1_statistics: Audio statistics for the first item.
        audio2_statistics: Audio statistics for the second item.
        video1_statistics: Video statistics for the first item.
        video2_statistics: Video statistics for the second item.
        unique_pairs: Number of unique (item1, item2) pairs.
    """

    text1_statistics: TextStatistics | None
    text2_statistics: TextStatistics | None
    image1_statistics: ImageStatistics | None
    image2_statistics: ImageStatistics | None
    audio1_statistics: AudioStatistics | None
    audio2_statistics: AudioStatistics | None
    video1_statistics: VideoStatistics | None
    video2_statistics: VideoStatistics | None
    unique_pairs: int
