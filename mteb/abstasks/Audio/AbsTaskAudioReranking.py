from __future__ import annotations

from typing import Any

from datasets import Dataset

from mteb.encoder_interface import AudioEncoder
from mteb.load_results.task_results import ScoresDict

from ...evaluation.evaluators.Audio.AudioRerankingEvaluator import (
    AudioRerankingEvaluator,
)
from ..AbsTask import AbsTask
from ..TaskMetadata import DescriptiveStatistics


class AudioRerankingDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Audio Reranking

    Attributes:
        num_samples: number of samples in the dataset.
        num_positive: Number of positive examples
        num_negative: Number of negative examples
        unique_query: Number of unique queries
        unique_positive: Number of unique positive examples
        unique_negative: Number of unique negative examples
    """

    num_samples: int
    num_positive: int
    num_negative: int
    unique_query: int
    unique_positive: int
    unique_negative: int


class AbsTaskAudioReranking(AbsTask):
    """Abstract class for audio re-ranking experiments.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
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
        model: AudioEncoder,
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
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        scores = evaluator(model)

        self._add_main_score(scores)
        return scores

    def _add_main_score(self, scores: ScoresDict) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
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

        return AudioRerankingDescriptiveStatistics(
            num_samples=len(query),
            num_positive=len(positive),
            num_negative=len(negative),
            unique_query=len(
                {str(q) for q in query}
            ),  # Use string representation for uniqueness
            unique_positive=len({str(p) for p in positive}),
            unique_negative=len({str(n) for n in negative}),
        )


def transform_audio_reranking_data(data: list[list[Any]] | list[Any]) -> list[Any]:
    """Transforms a list of lists of audio objects into a list of audio objects"""
    if not isinstance(data[0], list):
        return data
    return [item for sublist in data for item in sublist]
