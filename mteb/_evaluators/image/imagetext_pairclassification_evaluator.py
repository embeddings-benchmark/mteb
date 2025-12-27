from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader

from mteb._create_dataloaders import (
    _transform_image_to_rgb,
)
from mteb._evaluators.evaluator import Evaluator
from mteb._requires_package import requires_image_dependencies
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.models_protocols import EncoderProtocol

if TYPE_CHECKING:
    from PIL.Image import Image


logger = logging.getLogger(__name__)


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        images: list[Image],
    ):
        self.images = images

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Image]:
        return {
            "image": self.images[idx],
        }

    @property
    def features(self) -> dict[str, Any]:
        # for correct wrapper handling
        return {"image": []}


class ImageTextPairClassificationEvaluator(Evaluator):
    """Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and dissimilar image caption pairs.

    The goal is to find the correct image for each caption and the correct caption for each image.
    This is done by computing the similarities between each image and each caption.
    The results are written in a CSV. If a CSV already exists, then values are appended.
    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        images: Each row is a list of images.
        texts: Each row is a list of captions.
        batch_size: Batch size used to compute embeddings
    """

    def __init__(
        self,
        dataset,
        images_column_names: str | Sequence[str],
        texts_column_names: str | Sequence[str],
        num_images_per_sample: int,
        num_texts_per_sample: int,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        requires_image_dependencies()

        self.dataset = dataset
        self.images_column_names = images_column_names
        self.texts_column_names = texts_column_names
        self.num_images_per_sample = num_images_per_sample
        self.num_texts_per_sample = num_texts_per_sample
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(  # type: ignore[override]
        self, model: EncoderProtocol, *, encode_kwargs: dict[str, Any]
    ) -> list[torch.Tensor]:
        images = []
        if isinstance(self.images_column_names, str):
            images = self.dataset[self.images_column_names]
        else:
            for row in self.dataset:
                for col in self.images_column_names:
                    images.append(row[col])

        images = [_transform_image_to_rgb(img) for img in images]

        texts = []
        if isinstance(self.texts_column_names, str):
            texts = self.dataset[self.texts_column_names]
        else:
            for row in self.dataset:
                for col in self.texts_column_names:
                    texts.append(row[col])

        text_embeddings = model.encode(
            DataLoader(
                Dataset.from_dict({"text": texts}),
                **encode_kwargs,
            ),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        if not isinstance(text_embeddings, torch.Tensor):
            text_embeddings = torch.tensor(text_embeddings)

        norm_text_embeddings = F.normalize(
            text_embeddings,
            dim=-1,
        ).view(len(self.dataset), self.num_texts_per_sample, -1)

        image_embeddings = model.encode(
            DataLoader(
                CustomImageDataset(images),
                collate_fn=lambda x: {"image": [item["image"] for item in x]},
                **encode_kwargs,
            ),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )
        if not isinstance(image_embeddings, torch.Tensor):
            image_embeddings = torch.tensor(image_embeddings)

        norm_image_embeddings = F.normalize(
            image_embeddings,
            dim=-1,
        ).view(len(self.dataset), self.num_images_per_sample, -1)

        all_scores = []

        for img_emb, txt_emb in zip(norm_image_embeddings, norm_text_embeddings):
            scores = (
                img_emb @ txt_emb.t()
            )  # shape = (num_images_per_sample x num_texts_per_sample)
            all_scores.append(scores)
        return all_scores
