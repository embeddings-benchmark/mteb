from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from datasets import Dataset

from ...encoder_interface import Encoder
from ...evaluation.evaluators import AudioZeroshotClassificationEvaluator
from ..AbsTask import AbsTask, ScoresDict

logger = logging.getLogger(__name__)


class AbsTaskAudioZeroshotClassification(AbsTask):
    """Abstract class for ZeroshotClassification tasks
    The similarity between audio and candidate text prompts, such as as an audio wav of a dog barking and candidate text prompts like "Sound of a dog barking" or "Sound of a airplane".

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        image: list of Image.Image
        labels: list of int
    """

    audio_column_name: str = "audio"
    label_column_name: str = "target"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        pass

    def _undersample_data(self, dataset: Dataset) -> Dataset:
        """Undersample dataset to have samples_per_label samples for each numeric label"""
        labels = dataset[self.label_column_name]

        # Create label to index mapping (using numeric labels directly)
        label_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_indices[label].append(idx)

        # Sample indices
        selected_indices = []
        for label_num in sorted(label_to_indices.keys()):
            indices = label_to_indices[label_num]
            sample_size = min(self.samples_per_label, len(indices))
            sampled = np.random.choice(
                indices, size=sample_size, replace=False
            ).tolist()
            selected_indices.extend(sampled)

        logger.info(
            f"Subsampled from {len(dataset)} to {len(selected_indices)} samples"
        )

        return dataset.select(selected_indices)

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        candidate_labels = self.get_candidate_labels()

        if len(dataset) > 2048:
            dataset = self._undersample_data(dataset)

        evaluator = AudioZeroshotClassificationEvaluator(
            dataset,
            self.audio_column_name,
            self.label_column_name,
            candidate_labels,
            task_name=self.metadata.name,
            **kwargs,
        )
        metrics = evaluator(model, encode_kwargs=encode_kwargs)

        scores = {
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
            "f1_weighted": metrics["f1_weighted"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        self._add_main_score(scores)
        return scores

    def get_candidate_labels(self) -> list[str]:
        """Return the text candidates for zeroshot classification"""
        raise NotImplementedError("This method should be overridden by subclasses")
