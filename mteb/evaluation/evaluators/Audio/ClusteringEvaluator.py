from __future__ import annotations

import logging
import math
import os
from typing import Any

import sklearn
import sklearn.cluster
import torch
from datasets import Audio
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from torch.utils.data import DataLoader

from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.dataset_utils import AudioDataset, CustomAudioCollate
from mteb.evaluation.evaluators.Evaluator import Evaluator

logger = logging.getLogger(__name__)


class AudioClusteringEvaluator(Evaluator):
    def __init__(
        self,
        audio: list[Audio],
        labels: list[int],
        task_name: str | None = None,
        clustering_batch_size: int = 500,
        limit: int | None = None,
        model_sampling_rate: int | None = None,  # Added to get sampling rate earlier
        model_max_audio_length_s: float
        | None = None,  # Added to get max length earlier
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit is not None:
            audio = audio[:limit]
            labels = labels[:limit]
        self.audio = audio
        self.labels = labels
        self.clustering_batch_size = clustering_batch_size
        self.task_name = task_name

        self.model_sampling_rate = (
            model_sampling_rate if model_sampling_rate is not None else 16000
        )
        self.model_max_audio_length_s = (
            model_max_audio_length_s if model_max_audio_length_s is not None else 30.0
        )

    def __call__(self, model: Encoder, *, encode_kwargs: dict[str, Any] = {}):
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32

        # Get model-specific parameters for collate_fn - now from self
        # model_sampling_rate = getattr(model, "sampling_rate", 16000)  # Default if not explicitly set
        # model_max_audio_length_s = getattr(model, "max_audio_length_s", 30.0) # Default if not explicitly set
        max_length_samples_for_collate = int(
            self.model_max_audio_length_s * self.model_sampling_rate
        )

        audio_dataset = AudioDataset(
            hf_dataset=self.audio,
            target_sampling_rate=self.model_sampling_rate,
            mono=True,
        )
        audio_dataloader = DataLoader(
            audio_dataset,
            batch_size=encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=CustomAudioCollate(
                max_length_samples=max_length_samples_for_collate, pad_value=0.0
            ),
            num_workers=min(math.floor(os.cpu_count() / 2), 16),
        )

        # audio_embeddings = model.get_audio_embeddings(
        #     self.audio,
        #     batch_size=encode_kwargs["batch_size"],
        # )
        audio_embeddings_list = []
        for batch_data in audio_dataloader:
            batch_waveforms = batch_data["waveforms"].to(model.device)
            batch_embeddings = model.get_audio_embeddings(
                batch_waveforms,
                task_name=self.task_name,
                **encode_kwargs,
            )
            audio_embeddings_list.append(batch_embeddings)
        audio_embeddings = torch.cat(audio_embeddings_list, dim=0).cpu().numpy()

        logger.info("Fitting Mini-Batch K-Means model...")
        clustering_model = sklearn.cluster.MiniBatchKMeans(
            n_clusters=len(set(self.labels)),
            batch_size=self.clustering_batch_size,
            n_init="auto",
        )
        clustering_model.fit(audio_embeddings)
        cluster_assignment = clustering_model.labels_

        logger.info("Evaluating...")
        v_measure = metrics.cluster.v_measure_score(self.labels, cluster_assignment)
        nmi = metrics.cluster.normalized_mutual_info_score(
            self.labels, cluster_assignment
        )
        ari = metrics.cluster.adjusted_rand_score(self.labels, cluster_assignment)

        matrix = metrics.confusion_matrix(self.labels, cluster_assignment)

        # get linear sum assignment
        row_ind, col_ind = linear_sum_assignment(matrix, maximize=True)
        total_correct = matrix[row_ind, col_ind].sum()
        clustering_accuracy = total_correct / len(self.labels)

        return {
            "v_measure": v_measure,
            "nmi": nmi,
            "ari": ari,
            "cluster_accuracy": clustering_accuracy,
        }
