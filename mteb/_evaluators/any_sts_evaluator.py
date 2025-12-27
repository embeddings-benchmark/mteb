import logging
from typing import Any, TypedDict

from datasets import Dataset
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb._create_dataloaders import create_dataloader
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models import EncoderProtocol
from mteb.similarity_functions import compute_pairwise_similarity
from mteb.types import PromptType

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
        input1_prompt_type: PromptType | None,
        input2_prompt_type: PromptType | None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset
        self.input_columns = sentences_column_names
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset
        self.input1_prompt_type = input1_prompt_type
        self.input2_prompt_type = input2_prompt_type

    def __call__(
        self, model: EncoderProtocol, *, encode_kwargs: dict[str, Any]
    ) -> STSEvaluatorScores:
        logger.info("Running semantic similarity - Encoding samples (1/2)")
        embeddings1 = model.encode(
            create_dataloader(
                self.dataset,
                self.task_metadata,
                input_column=self.input_columns[0],
                **encode_kwargs,
            ),
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            prompt_type=self.input1_prompt_type,
            **encode_kwargs,
        )

        logger.info("Running semantic similarity - Encoding samples (2/2)...")
        embeddings2 = model.encode(
            create_dataloader(
                self.dataset,
                self.task_metadata,
                input_column=self.input_columns[1],
                **encode_kwargs,
            ),
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            prompt_type=self.input2_prompt_type,
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
