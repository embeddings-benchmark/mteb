from typing import Any

from datasets import Dataset

from mteb.abstasks import AbsTask
from mteb.evaluation.evaluators.audio.audio_reranking_evaluator import (
    AudioRerankingEvaluator,
)
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import ScoresDict
from mteb.types.statistics import DescriptiveStatistics


class AudioRerankingDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Audio Reranking

    Attributes:
        num_samples: number of samples in the dataset.
        total_audio_duration: Total duration of all audio in seconds.
        num_positive: Number of positive examples
        num_negative: Number of negative examples

        min_query_duration: Minimum duration of query audio in seconds
        avg_query_duration: Average duration of query audio in seconds
        max_query_duration: Maximum duration of query audio in seconds
        unique_query: Number of unique queries

        min_positive_duration: Minimum duration of positive examples in seconds
        avg_positive_duration: Average duration of positive examples in seconds
        max_positive_duration: Maximum duration of positive examples in seconds
        unique_positive: Number of unique positive examples

        min_negative_duration: Minimum duration of negative examples in seconds
        avg_negative_duration: Average duration of negative examples in seconds
        max_negative_duration: Maximum duration of negative examples in seconds
        unique_negative: Number of unique negative examples
    """

    num_samples: int
    total_audio_duration: float
    num_positive: int
    num_negative: int

    min_query_duration: float
    avg_query_duration: float
    max_query_duration: float
    unique_query: int

    min_positive_duration: float
    avg_positive_duration: float
    max_positive_duration: float
    unique_positive: int

    min_negative_duration: float
    avg_negative_duration: float
    max_negative_duration: float
    unique_negative: int


class AbsTaskAudioReranking(AbsTask):
    """Abstract class for audio re-ranking experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
        query: Audio object with 'array' and 'sampling_rate' fields
        positive: list of Audio objects (relevant audio samples)
        negative: list of Audio objects (irrelevant audio samples)
    """

    abstask_prompt = "Retrieve audio based on user audio query."
    audio_query_column_name: str = "query"
    audio_positive_column_name: str = "positive"
    audio_negative_column_name: str = "negative"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
        data_split: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> ScoresDict:
        evaluator = AudioRerankingEvaluator(
            data_split,
            self.audio_query_column_name,
            self.audio_positive_column_name,
            self.audio_negative_column_name,
            task_metadata=self.metadata,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        scores = evaluator(model)

        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> AudioRerankingDescriptiveStatistics:
        if hf_subset:
            query = self.dataset[hf_subset][split][self.audio_query_column_name]
            positive = transform_audio_reranking_data(
                self.dataset[hf_subset][split][self.audio_positive_column_name]
            )
            negative = transform_audio_reranking_data(
                self.dataset[hf_subset][split][self.audio_negative_column_name]
            )
        elif compute_overall:
            query = []
            positive = []
            negative = []
            for hf_subset in self.metadata.eval_langs:
                query.extend(
                    self.dataset[hf_subset][split][self.audio_query_column_name]
                )
                positive.extend(
                    transform_audio_reranking_data(
                        self.dataset[hf_subset][split][self.audio_positive_column_name]
                    )
                )
                negative.extend(
                    transform_audio_reranking_data(
                        self.dataset[hf_subset][split][self.audio_negative_column_name]
                    )
                )
        else:
            query = self.dataset[split][self.audio_query_column_name]
            positive = transform_audio_reranking_data(
                self.dataset[split][self.audio_positive_column_name]
            )
            negative = transform_audio_reranking_data(
                self.dataset[split][self.audio_negative_column_name]
            )

        def get_audio_duration(audio_obj):
            """Calculate audio duration in seconds"""
            if (
                isinstance(audio_obj, dict)
                and "array" in audio_obj
                and "sampling_rate" in audio_obj
            ):
                return len(audio_obj["array"]) / audio_obj["sampling_rate"]
            return 0.0  # Default if duration can't be calculated

        query_durations = [get_audio_duration(q) for q in query]
        positive_durations = [get_audio_duration(p) for p in positive]
        negative_durations = [get_audio_duration(n) for n in negative]

        total_duration = (
            sum(query_durations) + sum(positive_durations) + sum(negative_durations)
        )

        return AudioRerankingDescriptiveStatistics(
            num_samples=len(query),
            total_audio_duration=total_duration,
            num_positive=len(positive),
            num_negative=len(negative),
            min_query_duration=min(query_durations) if query_durations else 0.0,
            avg_query_duration=sum(query_durations) / len(query_durations)
            if query_durations
            else 0.0,
            max_query_duration=max(query_durations) if query_durations else 0.0,
            unique_query=len(
                {str(q) for q in query}
            ),  # Use string representation for uniqueness
            min_positive_duration=min(positive_durations)
            if positive_durations
            else 0.0,
            avg_positive_duration=sum(positive_durations) / len(positive_durations)
            if positive_durations
            else 0.0,
            max_positive_duration=max(positive_durations)
            if positive_durations
            else 0.0,
            unique_positive=len({str(p) for p in positive}),
            min_negative_duration=min(negative_durations)
            if negative_durations
            else 0.0,
            avg_negative_duration=sum(negative_durations) / len(negative_durations)
            if negative_durations
            else 0.0,
            max_negative_duration=max(negative_durations)
            if negative_durations
            else 0.0,
            unique_negative=len({str(n) for n in negative}),
        )


def transform_audio_reranking_data(data: list[list[Any]] | list[Any]) -> list[Any]:
    """Transforms a list of lists of audio objects into a list of audio objects"""
    if not isinstance(data[0], list):
        return data
    return [item for sublist in data for item in sublist]
