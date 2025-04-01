from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from torch.utils.data import DataLoader

from mteb.requires_package import requires_image_dependencies

from ..Evaluator import Evaluator

logger = logging.getLogger(__name__)


def get_default_transform():
    requires_image_dependencies()
    from torchvision import transforms

    return transforms.Compose([transforms.PILToTensor()])


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, image_column_name: str = "image", transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.image_column_name = image_column_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset[idx][self.image_column_name]
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        return image


def custom_collate_fn(batch):
    return batch


class VisualSTSEvaluator(Evaluator):
    def __init__(
        self,
        dataset,
        sentences_column_names: list[str],
        gold_scores: list[float],
        task_name: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        default_transform = get_default_transform()
        self.sentence1_dataset = ImageDataset(
            dataset,
            image_column_name=sentences_column_names[0],
            transform=default_transform,
        )
        self.sentence2_dataset = ImageDataset(
            dataset,
            image_column_name=sentences_column_names[1],
            transform=default_transform,
        )
        self.gold_scores = gold_scores
        self.task_name = task_name
        # TODO use task_name for prompts with interleaved encoding.

    def __call__(
        self,
        model,  # TODO: model type
        *,
        encode_kwargs: dict[str, Any] = {},
    ):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        sentence1_dataloader = DataLoader(
            self.sentence1_dataset,
            batch_size=encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )
        sentence2_dataloader = DataLoader(
            self.sentence2_dataset,
            batch_size=encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )

        embeddings1 = model.get_image_embeddings(
            sentence1_dataloader, batch_size=encode_kwargs["batch_size"]
        )
        embeddings2 = model.get_image_embeddings(
            sentence2_dataloader, batch_size=encode_kwargs["batch_size"]
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

        similarity_scores = None
        if hasattr(model, "similarity_pairwise"):
            similarity_scores = model.similarity_pairwise(embeddings1, embeddings2)  # type: ignore
        elif hasattr(model, "similarity"):
            _similarity_scores = [
                float(model.similarity(e1, e2))  # type: ignore
                for e1, e2 in zip(embeddings1, embeddings2)
            ]
            similarity_scores = np.array(_similarity_scores)

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
