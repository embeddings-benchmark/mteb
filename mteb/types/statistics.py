from __future__ import annotations

from typing import Generic, TypeVar

from typing_extensions import NotRequired, TypedDict

from mteb.types._result import HFSubset


class SplitDescriptiveStatistics(TypedDict):
    """Base class for descriptive statistics for the subset.

    Every per-task descriptive-stats TypedDict (Classification, Retrieval,
    STS, …) inherits from this. The per-task fields are added by each
    subclass; the multilingual subset wrapper is provided separately by
    :class:`DescriptiveStatistics`.
    """

    pass


# Self-bound type parameter for ``DescriptiveStatistics``: the API parameterises
# the generic wrapper with the per-task TypedDict so that ``hf_subset_descriptive_stats``
# is precisely typed as a mapping to the same per-task shape. The default keeps
# backward compatibility with existing callers that use ``DescriptiveStatistics``
# without parameters.
_DescStatsT = TypeVar(
    "_DescStatsT",
    bound="SplitDescriptiveStatistics",
    default="SplitDescriptiveStatistics",
)


class DescriptiveStatistics(SplitDescriptiveStatistics, Generic[_DescStatsT]):
    """Generic descriptive statistics for the full task (split + multilingual wrapper).

    Parameterised on the per-task ``SplitDescriptiveStatistics`` subclass so
    ``hf_subset_descriptive_stats`` carries the right per-task shape (e.g.
    ``DescriptiveStatistics[ClassificationDescriptiveStatistics]``).

    Attributes:
        num_samples: Total number of samples
        hf_subset_descriptive_stats: HFSubset descriptive statistics (only for multilingual datasets)
    """

    num_samples: int
    hf_subset_descriptive_stats: NotRequired[dict[HFSubset, _DescStatsT]]


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


# --- Per-task descriptive statistics -----------------------------------------
# Each concrete ``AbsTask._calculate_descriptive_statistics_from_split`` returns
# one of these per-split TypedDicts. They inherit directly from
# ``SplitDescriptiveStatistics`` — no ``hf_subset_descriptive_stats`` field —
# so ``**modality_stats`` spread in the constructor doesn't trip mypy's
# NotRequired-key check.
#
# Multilingual layering is expressed via ``DescriptiveStatistics[XxxDescriptiveStatistics]``
# at the API layer: the generic wrapper carries ``num_samples`` + a
# ``hf_subset_descriptive_stats`` whose values are precisely typed.


class AnySTSDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for STS.

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of unique pairs

        text1_statistics: Statistics for sentence1
        text2_statistics: Statistics for sentence2

        image1_statistics: Statistics for image1
        image2_statistics: Statistics for image2

        audio1_statistics: Statistics for audio1
        audio2_statistics: Statistics for audio2

        video1_statistics: Statistics for video1
        video2_statistics: Statistics for video2

        label_statistics: Statistics for labels
    """

    num_samples: int
    number_of_characters: int | None
    unique_pairs: int | None

    text1_statistics: TextStatistics | None
    text2_statistics: TextStatistics | None

    image1_statistics: ImageStatistics | None
    image2_statistics: ImageStatistics | None

    audio1_statistics: AudioStatistics | None
    audio2_statistics: AudioStatistics | None

    video1_statistics: VideoStatistics | None
    video2_statistics: VideoStatistics | None

    label_statistics: ScoreStatistics


class BitextDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Bitext.

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of duplicate pairs

        sentence1_statistics: Statistics for sentence1
        sentence2_statistics: Statistics for sentence2
    """

    num_samples: int
    number_of_characters: int
    unique_pairs: int

    sentence1_statistics: TextStatistics
    sentence2_statistics: TextStatistics


class ClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Classification.

    Attributes:
        num_samples: number of samples in the dataset.
        samples_in_train: Number of unique test samples (across all input modalities)
            that also appear in the train split. None when evaluated on the train split itself.

        text_statistics: Statistics for text
        image_statistics: Statistics for images
        audio_statistics: Statistics for audio
        video_statistics: Statistics for video
        label_statistics: Statistics for labels
    """

    num_samples: int
    samples_in_train: int | None

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    audio_statistics: AudioStatistics | None
    video_statistics: VideoStatistics | None
    label_statistics: LabelStatistics


class RegressionDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Regression.

    Attributes:
        num_samples: number of samples in the dataset.
        samples_in_train: Number of texts in the train split

        text_statistics: Statistics of texts
        image_statistics: Statistics of images
        audio_statistics: Statistics of audio
        video_statistics: Statistics of video

        values_statistics: Statistics of values
    """

    num_samples: int
    samples_in_train: int | None

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    audio_statistics: AudioStatistics | None
    video_statistics: VideoStatistics | None
    values_statistics: ScoreStatistics


class ClusteringDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Clustering (legacy AbsTaskClusteringLegacy).

    Attributes:
        num_samples: number of samples in the dataset.

        text_statistics: Statistics for text
        image_statistics: Statistics for images
        audio_statistics: Statistics for audio
        video_statistics: Statistics for video
        label_statistics: Statistics for labels
    """

    num_samples: int

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    audio_statistics: AudioStatistics | None
    video_statistics: VideoStatistics | None
    label_statistics: LabelStatistics


class ClusteringFastDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for ClusteringFast.

    Attributes:
        num_samples: number of samples in the dataset.

        text_statistics: Statistics for text
        image_statistics: Statistics for images
        audio_statistics: Statistics for audio
        video_statistics: Statistics for video
        labels_statistics: Statistics for labels
    """

    num_samples: int

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    audio_statistics: AudioStatistics | None
    video_statistics: VideoStatistics | None
    labels_statistics: LabelStatistics


class PairClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for PairClassification.

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.
        unique_pairs: Number of unique pairs

        text1_statistics: Statistics for sentence1
        image1_statistics: Statistics for image1
        audio1_statistics: Statistics for audio1

        text2_statistics: Statistics for sentence2
        image2_statistics: Statistics for image2
        audio2_statistics: Statistics for audio2

        labels_statistics: Statistics for labels
    """

    num_samples: int
    number_of_characters: int | None
    unique_pairs: int | None

    text1_statistics: TextStatistics | None
    image1_statistics: ImageStatistics | None
    audio1_statistics: AudioStatistics | None
    video1_statistics: VideoStatistics | None
    text2_statistics: TextStatistics | None
    image2_statistics: ImageStatistics | None
    audio2_statistics: AudioStatistics | None
    video2_statistics: VideoStatistics | None
    labels_statistics: LabelStatistics


class ZeroShotClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for ZeroShotClassification.

    Attributes:
        num_samples: number of samples in the dataset.

        text_statistics: Statistics for texts
        image_statistics: Statistics for images
        audio_statistics: Statistics for audio
        video_statistics: Statistics for video
        label_statistics: Statistics for dataset labels

        candidates_labels_text_statistics: Statistics for candidate labels text
    """

    num_samples: int

    text_statistics: TextStatistics | None
    image_statistics: ImageStatistics | None
    audio_statistics: AudioStatistics | None
    video_statistics: VideoStatistics | None
    label_statistics: LabelStatistics
    candidates_labels_text_statistics: TextStatistics


class RetrievalDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Retrieval.

    Attributes:
        num_samples: Total number of queries and documents
        num_queries: Number of queries
        num_documents: Number of documents
        number_of_characters: Total number of characters in queries and documents

        documents_text_statistics: Statistics for documents
        documents_image_statistics: Statistics for documents
        documents_audio_statistics: Statistics for documents
        documents_video_statistics: Statistics for documents
        queries_text_statistics: Statistics for queries
        queries_image_statistics: Statistics for queries
        queries_audio_statistics: Statistics for queries
        queries_video_statistics: Statistics for queries
        relevant_docs_statistics: Statistics for relevant documents
        top_ranked_statistics: Statistics for top ranked documents (if available)
    """

    num_samples: int
    num_queries: int
    num_documents: int
    number_of_characters: int

    documents_text_statistics: TextStatistics | None
    documents_image_statistics: ImageStatistics | None
    documents_audio_statistics: AudioStatistics | None
    documents_video_statistics: VideoStatistics | None

    queries_text_statistics: TextStatistics | None
    queries_image_statistics: ImageStatistics | None
    queries_audio_statistics: AudioStatistics | None
    queries_video_statistics: VideoStatistics | None

    relevant_docs_statistics: RelevantDocsStatistics

    # this is for datasets that do reranking
    top_ranked_statistics: TopRankedStatistics | None


class SummarizationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for Summarization.

    Attributes:
        num_samples: number of samples in the dataset.
        number_of_characters: Total number of symbols in the dataset.

        text_statistics: Statistics for the text
        human_summaries_statistics: Statistics for human summaries
        machine_summaries_statistics: Statistics for machine summaries
        score_statistics: Statistics for the relevance scores
    """

    num_samples: int
    number_of_characters: int

    text_statistics: TextStatistics
    human_summaries_statistics: TextStatistics
    machine_summaries_statistics: TextStatistics
    score_statistics: ScoreStatistics


class ImageTextPairClassificationDescriptiveStatistics(SplitDescriptiveStatistics):
    """Descriptive statistics for ImageTextPairClassification.

    Attributes:
        num_samples: number of samples in the dataset.
        text_statistics: Statistics for text
        image_statistics: Statistics for images
    """

    num_samples: int
    text_statistics: TextStatistics
    image_statistics: ImageStatistics
