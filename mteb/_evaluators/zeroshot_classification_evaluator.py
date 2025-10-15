import logging
from typing import Any

from datasets import Dataset

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.create_dataloaders import (
    create_dataloader,
    create_dataloader_from_texts,
)
from mteb.models import EncoderProtocol
from mteb.similarity_functions import similarity
from mteb.types import Array

from .evaluator import Evaluator

logger = logging.getLogger(__name__)


class ZeroShotClassificationEvaluator(Evaluator):
    def __init__(
        self,
        dataset: Dataset,
        input_column_name: str,
        candidate_labels: list[str],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.dataset = dataset
        self.input_column_name = input_column_name
        self.candidate_labels = candidate_labels
        self.task_metadata = task_metadata
        self.hf_split = hf_split
        self.hf_subset = hf_subset

    def __call__(
        self, model: EncoderProtocol, *, encode_kwargs: dict[str, Any]
    ) -> Array:
        dataloader = create_dataloader(
            self.dataset,
            batch_size=encode_kwargs["batch_size"],
            input_column=self.input_column_name,
            task_metadata=self.task_metadata,
        )

        logger.info("Running zero-shot classification - Encoding labels...")
        text_label_embeddings = model.encode(
            create_dataloader_from_texts(self.candidate_labels),
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        logger.info("Running zero-shot classification - Encoding samples...")
        input_embeddings = model.encode(
            dataloader,
            task_metadata=self.task_metadata,
            hf_subset=self.hf_subset,
            hf_split=self.hf_split,
            **encode_kwargs,
        )

        logger.info("Running zero-shot classification - Evaluating accuracy...")

        if self.task_metadata.modalities == ["text"]:
            probs = model.similarity(text_label_embeddings, input_embeddings)
        else:
            probs = similarity(text_label_embeddings, input_embeddings)
        return probs
