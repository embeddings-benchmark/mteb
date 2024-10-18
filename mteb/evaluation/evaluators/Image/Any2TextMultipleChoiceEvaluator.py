from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import transforms
from tqdm import tqdm

from mteb.encoder_interface import Encoder, EncoderWithSimilarity
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)

transform = transforms.Compose([transforms.PILToTensor()])


class Any2TextMultipleChoiceEvaluator(Evaluator):
    """Evaluate a model based on the similarity of queries (can be interleaved) and candidate answers.
    The goal is to find the correct text in multiple candidates that
    forms the correct answer of the interleaved query.

    Args:
        query_modalities: the modality of queries; supports image and text or either at the moment,
        query_column_names: column names of queries; should align with query modalities.
        label_column_name: column name of labels;
        choices_column_names: column name of candidate choices;
    """

    def __init__(
        self,
        dataset,
        query_modalities: str | list[str],
        query_column_names: dict,
        label_column_name: str,
        choices_column_name: str,
        task_name: str | None = None,
        transform=None,
        limit: int | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            dataset = dataset.select(range(limit))
        self.dataset = dataset
        self.query_modalities = query_modalities
        self.query_column_names = query_column_names
        self.label_column_name = label_column_name
        self.choices_column_name = choices_column_name
        self.task_name = task_name
        self.transform = transform

    def __call__(
        self,
        model: Encoder | EncoderWithSimilarity,
        encode_kwargs: dict[str, Any] = {},
    ):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 64

        label_list = list(
            {x for n in self.dataset[self.choices_column_name] for x in n}
        )
        label_embeddings = model.get_text_embeddings(label_list)
        label_embedding_dict = {}
        for label, embedding in zip(label_list, label_embeddings):
            label_embedding_dict[label] = embedding

        if "text" in self.query_modalities:
            questions = self.dataset[self.query_column_names["text"]]
        else:
            questions = None
        if "image" in self.query_modalities:
            images = self.dataset[self.query_column_names["image"]]
        query_embeddings = model.get_fused_embeddings(
            texts=questions,
            images=images,
            batch_size=encode_kwargs["batch_size"],
        )

        answers = self.dataset[self.label_column_name]
        choices = self.dataset[self.choices_column_name]

        # note that answers are the indeces
        predictions = []
        for q_embedding, choice in tqdm(zip(query_embeddings, choices)):
            choice_embeddings = torch.vstack(
                [label_embedding_dict[c] for c in choice]
            )  # (choice_size, embedding_dim)
            q_embedding = q_embedding[np.newaxis, :]
            cos_sim = cosine_similarity(q_embedding, choice_embeddings)
            predictions.append(np.argmax(cos_sim))

        metrics = {}
        metrics["accuracy"] = accuracy_score(predictions, answers)
        return metrics
