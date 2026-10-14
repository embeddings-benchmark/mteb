from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from datasets import Dataset

from mteb._evaluators import AnalogyEvaluator
from mteb.models import EncoderProtocol
from mteb.types.statistics import AnalogyDescriptiveStatistics

from ._statistics_calculation import calculate_single_input_modality_statistics
from .abstask import AbsTask

if TYPE_CHECKING:
    from pathlib import Path

    from mteb.models import MTEBModels
    from mteb.timing import TimingStack
    from mteb.types import EncodeKwargs

logger = logging.getLogger(__name__)


class AnalogyMetrics(TypedDict):
    """Metrics for Analogy tasks

    Attributes:
        hit_rate_at_1: Fraction of analogies whose answer is the most similar candidate.
        hit_rate_at_10: Fraction of analogies whose answer is among the 10 most similar candidates.
        avg_rank: Mean rank of the answer, where rank 1 is the most similar candidate.
        median_rank: Median rank of the answer.
    """

    hit_rate_at_1: float
    hit_rate_at_10: float
    avg_rank: float
    median_rank: float


class AbsTaskAnalogy(AbsTask):
    """Abstract class for analogy tasks (A is to B as C is to D) for any single modality.

    Each analogy names four concepts. Every concept is represented by one or more inputs (e.g. a word or several
    images), which are embedded and mean-pooled into a single L2-normalised concept embedding. For each analogy the
    concepts are ranked by cosine similarity to B - A + C, excluding A, B, and C.

    Attributes:
        dataset: Huggingface dataset with one analogy per row. Dataset must contain the columns in analogy_column_names,
            which hold concept names.
        analogy_column_names: Names of the columns holding A, B, C and D.
        concept_column_name: Name of the column of `get_concepts` holding the concept names.
        input_column_name: Name of the column of `get_concepts` holding the inputs that represent each concept.
        abstask_prompt: Prompt to use for the task for instruction model if not prompt is provided in TaskMetadata.prompt.
    """

    analogy_column_names: tuple[str, str, str, str] = (
        "word1",
        "word2",
        "word3",
        "word4",
    )
    concept_column_name: str = "concept"
    input_column_name: str = "text"
    abstask_prompt = "Represent the concept."

    def get_concepts(self, analogies: Dataset, split: str) -> Dataset:
        """Return the inputs representing each concept, as a dataset with a concept and an input column.

        A concept can have several inputs, whose embeddings are mean-pooled. By default each concept is represented
        by its own name as text. Override this to represent concepts in another way, e.g. by images.

        Args:
            analogies: The analogies of the split.
            split: The name of the split.
        """
        words = sorted(
            {word for column in self.analogy_column_names for word in analogies[column]}
        )
        return Dataset.from_dict(
            {self.concept_column_name: words, self.input_column_name: words}
        )

    def _evaluate_subset(
        self,
        model: MTEBModels,
        data_split: Dataset,
        *,
        hf_split: str,
        hf_subset: str,
        encode_kwargs: EncodeKwargs,
        prediction_folder: Path | None = None,
        num_proc: int | None = None,
        timer: TimingStack,
        **kwargs: Any,
    ) -> AnalogyMetrics:
        if not isinstance(model, EncoderProtocol):
            raise TypeError("Expected model to be an instance of EncoderProtocol")

        concepts = self.get_concepts(data_split, hf_split)
        missing = {
            word for column in self.analogy_column_names for word in data_split[column]
        } - set(concepts[self.concept_column_name])
        if missing:
            raise ValueError(
                f"Analogies use concepts without inputs: {sorted(missing)[:10]}"
            )

        evaluator = AnalogyEvaluator(
            data_split,
            concepts,
            analogy_column_names=self.analogy_column_names,
            concept_column_name=self.concept_column_name,
            input_column_name=self.input_column_name,
            task_metadata=self.metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            timer=timer,
            **kwargs,
        )
        ranks = evaluator(model, encode_kwargs=encode_kwargs, num_proc=num_proc)

        if prediction_folder:
            self._save_task_predictions(
                ranks,
                model,
                prediction_folder,
                hf_subset=hf_subset,
                hf_split=hf_split,
            )

        return self._calculate_scores(ranks)

    @staticmethod
    def _calculate_scores(ranks: list[int]) -> AnalogyMetrics:
        rank_array = np.array(ranks)
        return AnalogyMetrics(
            hit_rate_at_1=float(np.mean(rank_array == 1)),
            hit_rate_at_10=float(np.mean(rank_array <= 10)),
            avg_rank=float(np.mean(rank_array)),
            median_rank=float(np.median(rank_array)),
        )

    def _calculate_descriptive_statistics_from_split(
        self,
        split: str,
        *,
        hf_subset: str | None = None,
        compute_overall: bool = False,
        num_proc: int | None = None,
    ) -> AnalogyDescriptiveStatistics:
        if self.dataset is None:
            raise ValueError("Dataset is not loaded.")
        if hf_subset or compute_overall:
            raise NotImplementedError(
                "Descriptive statistics are only implemented for tasks with a single subset."
            )
        analogies: Dataset = self.dataset[split]
        concepts = self.get_concepts(analogies, split)

        documents_per_concept = np.unique(
            concepts[self.concept_column_name], return_counts=True
        )[1]
        modality_stats = calculate_single_input_modality_statistics(
            {self.metadata.modalities[0]: concepts[self.input_column_name]},
            max_workers=num_proc,
        )
        text_statistics = modality_stats["text_statistics"]
        return AnalogyDescriptiveStatistics(
            num_samples=len(analogies),
            num_concepts=len(documents_per_concept),
            num_documents=len(concepts),
            number_of_characters=(
                text_statistics["total_text_length"] if text_statistics else None
            ),
            min_documents_per_concept=int(documents_per_concept.min()),
            average_documents_per_concept=float(documents_per_concept.mean()),
            max_documents_per_concept=int(documents_per_concept.max()),
            documents_text_statistics=text_statistics,
            documents_image_statistics=modality_stats["image_statistics"],
            documents_audio_statistics=modality_stats["audio_statistics"],
            documents_video_statistics=modality_stats["video_statistics"],
        )

    def _push_dataset_to_hub(
        self,
        repo_name: str,
        num_proc: int | None = None,
        **kwargs: Any,
    ) -> None:
        self._upload_dataset_to_hub(
            repo_name, list(self.analogy_column_names), num_proc=num_proc, **kwargs
        )
