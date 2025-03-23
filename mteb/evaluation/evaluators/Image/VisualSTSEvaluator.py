from __future__ import annotations

import logging
from typing import Any

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)

from mteb.abstasks import TaskMetadata
from mteb.create_dataloaders import create_image_dataloader

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


class VisualSTSEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        sentences_column_names: list[str],
        gold_scores: list[float],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        task_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sentence1_dataset = create_image_dataloader(
            (
                dataset.select_columns(sentences_column_names[0]).rename_column(
                    sentences_column_names[0], "image"
                )
            ),
        )
        self.sentence2_dataset = create_image_dataloader(
            (
                dataset.select_columns(sentences_column_names[1]).rename_column(
                    sentences_column_names[1], "image"
                )
            ),
        )
        self.gold_scores = gold_scores
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self,
        model,  # TODO: model type
        *,
        encode_kwargs: dict[str, Any] = {},
    ):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        # self.sentence1_dataset.batch_size = encode_kwargs["batch_size"]
        # self.sentence2_dataset.batch_size = encode_kwargs["batch_size"]

        embeddings1 = model.encode(
            self.sentence1_dataset,
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            batch_size=encode_kwargs["batch_size"],
        )
        embeddings2 = model.encode(
            self.sentence2_dataset,
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            batch_size=encode_kwargs["batch_size"],
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
            pearson = pearsonr(self.gold_scores, similarity_scores)
            spearman = spearmanr(self.gold_scores, similarity_scores)
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
