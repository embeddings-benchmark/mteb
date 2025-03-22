from __future__ import annotations

import logging
from typing import Any

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.encoder_interface import Encoder

from ...create_dataloaders import create_dataloader_from_texts
from .Evaluator import Evaluator

logger = logging.getLogger(__name__)


class STSEvaluator(Evaluator):
    def __init__(
        self,
        sentences1,
        sentences2,
        gold_scores,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.gold_scores = gold_scores
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model: Encoder,
        *,
        encode_kwargs: dict[str, Any] = {},
    ):
        embeddings1 = model.encode(
            create_dataloader_from_texts(self.sentences1),
            task_metadata=self.task_metadata,
            hf_split=self.hf_split,
            hf_subset=self.hf_subset,
            **encode_kwargs,
        )
        embeddings2 = model.encode(
            create_dataloader_from_texts(self.sentences2),
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

        similarity_scores = model.similarity_pairwise(embeddings1, embeddings2)  # type: ignore

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
