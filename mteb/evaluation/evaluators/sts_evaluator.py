from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.encoder_interface import Encoder

from ...create_dataloaders import (
    create_dataloader,
)
from ...similarity_functions import compute_pairwise_similarity
from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class AnySTSEvaluator(Evaluator):  # TODO: Should we rename this to just STSEvaluator?
    def __init__(
        self,
        dataset: Dataset,
        sentences_column_names: tuple[str, str],
        gold_scores: list[float],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.first_column = create_dataloader(
            dataset,
            task_metadata,
            sentences_column_names[0],
        )
        self.second_column = create_dataloader(
            dataset,
            task_metadata,
            sentences_column_names[1],
        )
        self.gold_scores = gold_scores
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any],
    ):
        embeddings1 = model.encode(
            self.first_column,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        embeddings2 = model.encode(
            self.second_column,
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )

        logger.info("Evaluating...")
        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
        manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
        euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)

        cosine_pearson, _ = pearsonr(self.gold_scores, cosine_scores)
        cosine_spearman, _ = spearmanr(self.gold_scores, cosine_scores)

        manhatten_pearson, _ = pearsonr(self.gold_scores, manhattan_distances)
        manhatten_spearman, _ = spearmanr(self.gold_scores, manhattan_distances)

        euclidean_pearson, _ = pearsonr(self.gold_scores, euclidean_distances)
        euclidean_spearman, _ = spearmanr(self.gold_scores, euclidean_distances)

        similarity_scores = compute_pairwise_similarity(model, embeddings1, embeddings2)

        if similarity_scores is not None:
            pearson, _ = pearsonr(self.gold_scores, similarity_scores)
            spearman, _ = spearmanr(self.gold_scores, similarity_scores)
        else:
            # if model does not have a similarity function, we assume the cosine similarity
            pearson = cosine_pearson
            spearman = cosine_spearman

        return {
            # using the models own similarity score
            "pearson": pearson,
            "spearman": spearman,
            # generic similarity scores
            "cosine_pearson": cosine_pearson,
            "cosine_spearman": cosine_spearman,
            "manhattan_pearson": manhatten_pearson,
            "manhattan_spearman": manhatten_spearman,
            "euclidean_pearson": euclidean_pearson,
            "euclidean_spearman": euclidean_spearman,
        }
