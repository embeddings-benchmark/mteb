import logging
from typing import Any, TypedDict

from datasets import Dataset
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import create_dataloader
from mteb.models import Encoder

from ..similarity_functions import compute_pairwise_similarity
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class STSEvaluatorScores(TypedDict):
    """Scores for STS evaluation

    Attributes:
        cosine_scores: Cosine similarity scores between pairs of sentences.
        manhattan_distances: Negative Manhattan distances between pairs of sentences.
        euclidean_distances: Negative Euclidean distances between pairs of sentences.
        similarity_scores: Similarity scores computed using the model's similarity function, if available.
    """

    cosine_scores: list[float]
    manhattan_distances: list[float]
    euclidean_distances: list[float]
    similarity_scores: list[float] | None


class AnySTSEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        sentences_column_names: tuple[str, str],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.first_column = create_dataloader(
            dataset,
            task_metadata,
            input_column=sentences_column_names[0],
        )
        self.second_column = create_dataloader(
            dataset,
            task_metadata,
            input_column=sentences_column_names[1],
        )
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
    ) -> STSEvaluatorScores:
        logger.info("Running semantic similarity - Encoding samples (1/2)")
        embeddings1 = model.encode(
            self.first_column,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )

        logger.info("Running semantic similarity - Encoding samples (2/2)...")
        embeddings2 = model.encode(
            self.second_column,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )

        logger.info("Running semantic similarity - Evaluating similarity...")
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
        similarity_scores = compute_pairwise_similarity(model, embeddings1, embeddings2)

        logger.info("Running semantic similarity - Finished.")
        return STSEvaluatorScores(
            cosine_scores=cosine_scores.tolist(),
            manhattan_distances=manhattan_distances.tolist(),
            euclidean_distances=euclidean_distances.tolist(),
            similarity_scores=similarity_scores.tolist(),
        )
