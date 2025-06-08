from __future__ import annotations

import io
import logging
import math
import os
from typing import Any

import numpy as np
import torch
import torchaudio
import tqdm
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from mteb.encoder_interface import AudioEncoder, PromptType
from mteb.evaluation.evaluators.Evaluator import Evaluator
from mteb.evaluation.evaluators.utils import cos_sim, nAUC

logger = logging.getLogger(__name__)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, audio_samples, transform=None):
        self.audio_samples = audio_samples
        self.transform = transform or (lambda x: x)  # Identity transform by default

    def __len__(self):
        return len(self.audio_samples)

    def __getitem__(self, idx):
        audio = self.audio_samples[idx]

        # Handle different types of audio inputs
        if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
            # Already in the expected format with array and sampling_rate
            waveform = torch.tensor(audio["array"])
            sample_rate = audio["sampling_rate"]
        elif isinstance(audio, bytes):
            # Byte stream of audio data
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio))
        elif isinstance(audio, str):
            # Path to audio file
            waveform, sample_rate = torchaudio.load(audio)
        else:
            # Assume it's already a tensor or in a usable format
            waveform = audio

        if self.transform is not None:
            waveform = self.transform(waveform)

        return waveform


def custom_collate_fn(batch):
    return batch


class AudioRerankingEvaluator(Evaluator):
    """This class evaluates an AudioEncoder model for the task of audio re-ranking.
    Given an audio query and a list of audio documents, it computes the score [query, doc_i] for all possible
    documents and sorts them in decreasing order. Then, MRR@10 and MAP are computed to measure the quality of the ranking.

    The dataset should contain:
    - query: Audio object (query audio)
    - positive: List of Audio objects (relevant audio files)
    - negative: List of Audio objects (irrelevant audio files)
    """

    def __init__(
        self,
        samples,
        query_column_name: str = "query",
        positive_column_name: str = "positive",
        negative_column_name: str = "negative",
        task_name: str | None = None,
        mrr_at_k: int = 10,
        name: str = "",
        encode_kwargs: dict[str, Any] = {},
        use_batched_encoding: bool = True,
        limit: int | None = None,
        k_values: list[int] = [1, 3, 5, 10, 20, 100],
        transform=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if limit:
            samples = samples.train_test_split(limit)["test"]
        self.samples = samples
        self.query_column_name = query_column_name
        self.positive_column_name = positive_column_name
        self.negative_column_name = negative_column_name
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.use_batched_encoding = use_batched_encoding
        self.task_name = task_name
        self.k_values = k_values
        self.encode_kwargs = encode_kwargs
        self.transform = transform

        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32  # Smaller batch size for audio

        ### Remove samples with empty positive / negative set
        filtered_samples = []
        for sample in self.samples:
            if (
                len(sample[self.positive_column_name]) > 0
                and len(sample[self.negative_column_name]) > 0
            ):
                filtered_samples.append(sample)

        self.samples = filtered_samples

        if len(self.samples) == 0:
            raise ValueError(
                "No valid samples found with non-empty positive and negative sets"
            )

    def __call__(self, model: AudioEncoder):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model: AudioEncoder):
        return (
            self.compute_metrics_batched(model)
            if self.use_batched_encoding
            else self.compute_metrics_individual(model)
        )

    def compute_metrics_batched(self, model: AudioEncoder):
        """Computes the metrics in a batched way, by batching all queries and
        all documents together
        """
        logger.info("Encoding queries...")
        all_query_audios = [sample[self.query_column_name] for sample in self.samples]
        query_dataset = AudioDataset(all_query_audios, transform=self.transform)
        query_dataloader = DataLoader(
            query_dataset,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 4),
        )

        all_query_embs = np.asarray(
            model.encode(
                query_dataloader,
                task_name=self.task_name,
                prompt_type=PromptType.query,
                **self.encode_kwargs,
            )
        )

        all_mrr_scores = []
        all_ap_scores = []

        # Collect all documents
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample[self.positive_column_name])
            all_docs.extend(sample[self.negative_column_name])

        docs_dataset = AudioDataset(all_docs, transform=self.transform)
        docs_dataloader = DataLoader(
            docs_dataset,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=custom_collate_fn,
            num_workers=min(math.floor(os.cpu_count() / 2), 4),
        )

        all_docs_embs = np.asarray(
            model.encode(
                docs_dataloader,
                task_name=self.task_name,
                prompt_type=PromptType.passage,
                **self.encode_kwargs,
            )
        )

        # Compute scores
        logger.info("Evaluating...")
        docs_idx = 0
        for i, instance in enumerate(self.samples):
            query_emb = all_query_embs[i : i + 1]  # Single query embedding

            num_pos = len(instance[self.positive_column_name])
            num_neg = len(instance[self.negative_column_name])
            docs_emb = all_docs_embs[docs_idx : docs_idx + num_pos + num_neg]
            docs_idx += num_pos + num_neg

            if num_pos == 0 or num_neg == 0:
                continue

            is_relevant = [True] * num_pos + [False] * num_neg

            # Calculate similarity scores
            sim_scores = cos_sim(query_emb, docs_emb)
            sim_scores = sim_scores.cpu().numpy()[0]  # Flatten

            # Calculate metrics
            metrics = self._compute_metrics(sim_scores, is_relevant)
            all_mrr_scores.append(metrics["mrr"])
            all_ap_scores.append(metrics["map"])

        return self._collect_results(all_mrr_scores, all_ap_scores)

    def compute_metrics_individual(self, model: AudioEncoder):
        """Encodes and evaluates each (query, positive, negative) tuple individually.
        Slower but more memory efficient.
        """
        all_mrr_scores = []
        all_ap_scores = []

        for instance in tqdm.tqdm(self.samples, desc="Evaluating samples"):
            query_audio = instance[self.query_column_name]
            positive_audios = list(instance[self.positive_column_name])
            negative_audios = list(instance[self.negative_column_name])

            if len(positive_audios) == 0 or len(negative_audios) == 0:
                continue

            docs_audios = positive_audios + negative_audios
            is_relevant = [True] * len(positive_audios) + [False] * len(negative_audios)

            # Prepare audio for encoding
            query_dataset = AudioDataset([query_audio], transform=self.transform)
            query_dataloader = DataLoader(
                query_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=1,
            )

            docs_dataset = AudioDataset(docs_audios, transform=self.transform)
            docs_dataloader = DataLoader(
                docs_dataset,
                batch_size=self.encode_kwargs["batch_size"],
                shuffle=False,
                collate_fn=custom_collate_fn,
                num_workers=min(math.floor(os.cpu_count() / 2), 4),
            )

            # Encode query and documents
            query_emb = np.asarray(
                model.encode(
                    query_dataloader,
                    task_name=self.task_name,
                    prompt_type=PromptType.query,
                    **self.encode_kwargs,
                )
            )

            docs_emb = np.asarray(
                model.encode(
                    docs_dataloader,
                    task_name=self.task_name,
                    prompt_type=PromptType.passage,
                    **self.encode_kwargs,
                )
            )

            # Calculate similarity scores
            sim_scores = cos_sim(query_emb, docs_emb)
            sim_scores = sim_scores.cpu().numpy()[0]  # Flatten

            # Calculate metrics
            metrics = self._compute_metrics(sim_scores, is_relevant)
            all_mrr_scores.append(metrics["mrr"])
            all_ap_scores.append(metrics["map"])

        return self._collect_results(all_mrr_scores, all_ap_scores)

    def _collect_results(self, all_mrr_scores, all_ap_scores):
        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        results = {
            "map": mean_ap,
            f"mrr@{self.mrr_at_k}": mean_mrr,
        }

        # Add nAUC scores if we have multiple samples
        if len(all_mrr_scores) > 1:
            try:
                # Use ap_scores as confidence scores instead of mrr_scores
                all_mrr_scores_array = np.array(all_mrr_scores)
                all_ap_scores_array = np.array(all_ap_scores)

                # Check for potential NaN or inf values
                if (
                    np.isnan(all_ap_scores_array).any()
                    or np.isinf(all_ap_scores_array).any()
                ):
                    logger.warning(
                        "Found NaN or inf values in confidence scores, skipping nAUC calculation"
                    )
                else:
                    nAUC_values = nAUC(all_ap_scores_array, all_mrr_scores_array)
                    for k, v in nAUC_values.items():
                        results[k] = v
            except Exception as e:
                logger.warning(f"Error calculating nAUC: {e}")
                # Continue without nAUC values

        return results

    def _compute_metrics(self, sim_scores, is_relevant):
        """Compute Mean Average Precision and Mean Reciprocal Rank for a single query"""
        # Compute ranks and metrics
        sim_scores = torch.tensor(sim_scores)
        pred_scores = sim_scores.cpu().tolist()

        # Sort scores in descending order
        sorted_indices = np.argsort(-sim_scores)
        pred_ranking = [is_relevant[idx] for idx in sorted_indices]

        # Calculate MRR@k
        mrr = self._mrr_at_k_score(pred_ranking, self.mrr_at_k)

        # Calculate MAP (Mean Average Precision)
        ap = average_precision_score(is_relevant, pred_scores)

        return {"mrr": mrr, "map": ap}

    @staticmethod
    def _mrr_at_k_score(pred_ranking, k):
        """Calculate Mean Reciprocal Rank@k"""
        # Find the first position where a relevant document appears
        for i in range(min(k, len(pred_ranking))):
            if pred_ranking[i]:
                return 1.0 / (i + 1.0)
        return 0.0
