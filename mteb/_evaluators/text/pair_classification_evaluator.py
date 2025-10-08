from __future__ import annotations

import logging
from typing import Any, TypedDict

import numpy as np
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import create_dataloader_from_texts
from mteb.models import Encoder
from mteb.similarity_functions import compute_pairwise_similarity

from ..evaluator import Evaluator

logger = logging.getLogger(__name__)


class PairClassificationDistances(TypedDict):
    """Pair classification distances.

    Attributes:
        cosine_scores: Cosine similarity scores.
        euclidean_distances: Euclidean similarity scores.
        manhattan_distances: Manhattan similarity scores.
        similarity_scores: Similarity scores.
        dot_scores: Dot similarity scores.
    """

    cosine_scores: list[float]
    euclidean_distances: list[float]
    manhattan_distances: list[float]
    similarity_scores: list[float]
    dot_scores: list[float]


class PairClassificationEvaluator(Evaluator):
    """Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        sentences1: The first column of sentences
        sentences2: The second column of sentences
        labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
        batch_size: Batch size used to compute embeddings
    """

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

        if len(self.sentences1) != len(self.sentences2):
            raise ValueError(
                f"Sentences1 and Sentences2 must have the same length for task {task_metadata.name}"
            )

    def __call__(
        self,
        model: Encoder,
        encode_kwargs: dict[str, Any],
    ) -> PairClassificationDistances:
        all_sentences = self.sentences1[:] + self.sentences2[:]
        len_sentences1 = len(self.sentences1)
        embeddings = self._encode_unique_texts(
            all_sentences,
            model,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        embeddings1 = embeddings[:len_sentences1]
        embeddings2 = embeddings[len_sentences1:]

        logger.info("Computing similarity distances.")
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        manhattan_distances = paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = paired_euclidean_distances(embeddings1, embeddings2)

        similarity_scores = compute_pairwise_similarity(model, embeddings1, embeddings2)

        embeddings1_np = np.asarray(embeddings1)
        embeddings2_np = np.asarray(embeddings2)
        dot_scores = np.asarray(
            [
                np.dot(embeddings1_np[i], embeddings2_np[i])
                for i in range(len(embeddings1_np))
            ]
        )
        return PairClassificationDistances(
            cosine_scores=cosine_scores.tolist(),
            euclidean_distances=euclidean_distances.tolist(),
            manhattan_distances=manhattan_distances.tolist(),
            similarity_scores=similarity_scores.tolist(),
            dot_scores=dot_scores.tolist(),
        )

    @staticmethod
    def _encode_unique_texts(
        all_texts: list[str],
        model: Encoder,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **encode_kwargs: Any,
    ) -> np.ndarray:
        index_map, all_unique_texts, all_texts_indexes = {}, [], []
        for text in all_texts:
            text_hash = hash(text)
            if text_hash not in index_map:
                index_map[text_hash] = len(all_unique_texts)
                all_unique_texts.append(text)
            all_texts_indexes.append(index_map[text_hash])
        logger.warning(
            f"A total on {len(all_texts) - len(all_unique_texts)}/{len(all_texts)} duplicate texts were found during encoding. Only encoding unique text and duplicating embeddings across."
        )
        all_unique_texts_embs = np.asarray(
            model.encode(
                create_dataloader_from_texts(all_unique_texts),
                task_metadata=task_metadata,
                hf_split=hf_split,
                hf_subset=hf_subset,
                **encode_kwargs,
            )
        )
        return all_unique_texts_embs[all_texts_indexes]
