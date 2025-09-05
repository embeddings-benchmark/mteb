from __future__ import annotations

import logging
import math
import os
from typing import Any

import numpy as np
import torch
import tqdm
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from mteb.encoder_interface import AudioEncoder, PromptType
from mteb.evaluation.evaluators.dataset_utils import AudioDataset, CustomAudioCollate
from mteb.evaluation.evaluators.Evaluator import Evaluator
from mteb.evaluation.evaluators.utils import confidence_scores, cos_sim, nAUC

logger = logging.getLogger(__name__)


# def custom_collate_fn(batch):
#     return batch


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
        k_values: list[int] = [1, 3, 5, 10, 20, 100, 1000],
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

        # Get model-specific parameters for collate_fn
        model_sampling_rate = getattr(
            model, "sampling_rate", 16000
        )  # Default if not explicitly set
        model_max_audio_length_s = getattr(
            model, "max_audio_length_s", 30.0
        )  # Default if not explicitly set
        max_length_samples_for_collate = int(
            model_max_audio_length_s * model_sampling_rate
        )

        query_dataset = AudioDataset(
            hf_dataset=all_query_audios,
            audio_column_name=None,  # This tells AudioDataset to treat items as direct audio objects
            target_sampling_rate=model_sampling_rate,
            mono=True,
            transform=self.transform,
        )

        query_dataloader = DataLoader(
            query_dataset,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=CustomAudioCollate(
                max_length_samples=max_length_samples_for_collate, pad_value=0.0
            ),
            num_workers=min(math.floor(os.cpu_count() / 4), 4),
        )

        # all_query_embs = np.asarray(
        #     model.get_audio_embeddings(
        #         query_dataloader,
        #         task_name=self.task_name,
        #         prompt_type=PromptType.query,
        #         **self.encode_kwargs,
        #     )
        # )

        # New way to get audio embeddings, iterating over the dataloader and unpacking
        all_query_embs_list = []
        for batch_data in query_dataloader:
            batch_waveforms = batch_data["waveforms"].to(model.device)
            # Assuming model.get_audio_embeddings can take a batch of waveforms directly
            # If it expects a DataLoader, this part will need adjustment.
            batch_embeddings = model.get_audio_embeddings(
                batch_waveforms,
                task_name=self.task_name,
                prompt_type=PromptType.query,
                **self.encode_kwargs,
            )
            all_query_embs_list.append(batch_embeddings)
        all_query_embs = np.concatenate(all_query_embs_list, axis=0)

        all_mrr_scores = []
        all_ap_scores = []
        all_conf_scores = []

        # Collect all documents
        all_docs = []
        for sample in self.samples:
            all_docs.extend(sample[self.positive_column_name])
            all_docs.extend(sample[self.negative_column_name])

        docs_dataset = AudioDataset(
            hf_dataset=all_docs,
            audio_column_name=None,  # Same here
            target_sampling_rate=model_sampling_rate,
            mono=True,
            transform=self.transform,
        )
        docs_dataloader = DataLoader(
            docs_dataset,
            batch_size=self.encode_kwargs["batch_size"],
            shuffle=False,
            collate_fn=CustomAudioCollate(
                max_length_samples=max_length_samples_for_collate, pad_value=0.0
            ),
            num_workers=min(math.floor(os.cpu_count() / 4), 4),
        )

        # all_docs_embs = np.asarray(
        #     model.get_audio_embeddings(
        #         docs_dataloader,
        #         task_name=self.task_name,
        #         prompt_type=PromptType.document,
        #         **self.encode_kwargs,
        #     )
        # )

        # New way to get audio embeddings, iterating over the dataloader and unpacking
        all_docs_embs_list = []
        for batch_data in docs_dataloader:
            batch_waveforms = batch_data["waveforms"].to(model.device)
            batch_embeddings = model.get_audio_embeddings(
                batch_waveforms,
                task_name=self.task_name,
                prompt_type=PromptType.document,
                **self.encode_kwargs,
            )
            all_docs_embs_list.append(batch_embeddings)
        all_docs_embs = np.concatenate(all_docs_embs_list, axis=0)

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

            # Compute confidence scores properly
            conf_scores = confidence_scores(sim_scores)
            all_conf_scores.append(conf_scores)

            # Calculate metrics
            metrics = self._compute_metrics(sim_scores, is_relevant)
            all_mrr_scores.append(metrics["mrr"])
            all_ap_scores.append(metrics["map"])

        return self._collect_results(all_mrr_scores, all_ap_scores, all_conf_scores)

    def compute_metrics_individual(self, model: AudioEncoder):
        """Encodes and evaluates each (query, positive, negative) tuple individually.
        Slower but more memory efficient.
        """
        all_mrr_scores = []
        all_ap_scores = []
        all_conf_scores = []

        # Get model-specific parameters for collate_fn
        model_sampling_rate = getattr(
            model, "sampling_rate", 16000
        )  # Default if not explicitly set
        model_max_audio_length_s = getattr(
            model, "max_audio_length_s", 10.0
        )  # Default if not explicitly set
        max_length_samples_for_collate = int(
            model_max_audio_length_s * model_sampling_rate
        )

        for instance in tqdm.tqdm(self.samples, desc="Evaluating samples"):
            query_audio = instance[self.query_column_name]
            positive_audios = list(instance[self.positive_column_name])
            negative_audios = list(instance[self.negative_column_name])

            if len(positive_audios) == 0 or len(negative_audios) == 0:
                continue

            docs_audios = positive_audios + negative_audios
            is_relevant = [True] * len(positive_audios) + [False] * len(negative_audios)

            # Prepare audio for encoding
            query_dataset = AudioDataset(
                hf_dataset=[query_audio],
                target_sampling_rate=model_sampling_rate,
                mono=True,
                transform=self.transform,
            )
            query_dataloader = DataLoader(
                query_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=CustomAudioCollate(
                    max_length_samples=max_length_samples_for_collate, pad_value=0.0
                ),
                num_workers=1,
            )

            docs_dataset = AudioDataset(
                hf_dataset=docs_audios,
                target_sampling_rate=model_sampling_rate,
                mono=True,
                transform=self.transform,
            )
            docs_dataloader = DataLoader(
                docs_dataset,
                batch_size=self.encode_kwargs["batch_size"],
                shuffle=False,
                collate_fn=CustomAudioCollate(
                    max_length_samples=max_length_samples_for_collate, pad_value=0.0
                ),
                num_workers=min(math.floor(os.cpu_count() / 4), 4),
            )

            # Encode query and documents
            # query_emb = np.asarray(
            #     model.get_audio_embeddings(
            #         query_dataloader,
            #         task_name=self.task_name,
            #         prompt_type=PromptType.query,
            #         **self.encode_kwargs,
            #     )
            # )
            query_embs_list = []
            for batch_data in query_dataloader:
                batch_waveforms = batch_data["waveforms"].to(model.device)
                batch_embeddings = model.get_audio_embeddings(
                    batch_waveforms,
                    task_name=self.task_name,
                    prompt_type=PromptType.query,
                    **self.encode_kwargs,
                )
                query_embs_list.append(batch_embeddings)
            query_emb = np.concatenate(query_embs_list, axis=0)

            # docs_emb = np.asarray(
            #     model.get_audio_embeddings(
            #         docs_dataloader,
            #         task_name=self.task_name,
            #         prompt_type=PromptType.document,
            #         **self.encode_kwargs,
            #     )
            # )
            docs_embs_list = []
            for batch_data in docs_dataloader:
                batch_waveforms = batch_data["waveforms"].to(model.device)
                batch_embeddings = model.get_audio_embeddings(
                    batch_waveforms,
                    task_name=self.task_name,
                    prompt_type=PromptType.document,
                    **self.encode_kwargs,
                )
                docs_embs_list.append(batch_embeddings)
            docs_emb = np.concatenate(docs_embs_list, axis=0)

            # Calculate similarity scores
            sim_scores = cos_sim(query_emb, docs_emb)
            sim_scores = sim_scores.cpu().numpy()[0]  # Flatten

            # Compute confidence scores properly
            conf_scores = confidence_scores(sim_scores)
            all_conf_scores.append(conf_scores)

            # Calculate metrics
            metrics = self._compute_metrics(sim_scores, is_relevant)
            all_mrr_scores.append(metrics["mrr"])
            all_ap_scores.append(metrics["map"])

        return self._collect_results(all_mrr_scores, all_ap_scores, all_conf_scores)

    def _collect_results(self, all_mrr_scores, all_ap_scores, all_conf_scores):
        mean_ap = np.mean(all_ap_scores)
        mean_mrr = np.mean(all_mrr_scores)

        results = {
            "map": mean_ap,
            "mrr": mean_mrr,  # Add standard mrr key
            f"mrr@{self.mrr_at_k}": mean_mrr,
        }

        # Compute nAUCs properly like text reranking
        if len(all_mrr_scores) > 1 and len(all_conf_scores) > 0:
            try:
                naucs_map = self.nAUC_scores(all_conf_scores, all_ap_scores, "map")
                naucs_mrr = self.nAUC_scores(all_conf_scores, all_mrr_scores, "mrr")
                results.update(naucs_map)
                results.update(naucs_mrr)
            except Exception as e:
                logger.warning(f"Error calculating nAUC: {e}")

        return results

    @staticmethod
    def nAUC_scores(
        all_conf_scores: list[dict[str, float]],
        metrics: list[float],
        metric_name: str,
    ) -> dict[str, float]:
        """Computes normalized Area Under the Curve on a set of evaluated instances"""
        conf_fcts = list(all_conf_scores[0].keys())
        all_conf_scores = {
            fct: np.array([x[fct] for x in all_conf_scores]) for fct in conf_fcts
        }
        metrics = np.array(metrics)
        naucs = {
            f"nAUC_{metric_name}_{fct}": nAUC(all_conf_scores[fct], metrics)
            for fct in conf_fcts
        }
        return naucs

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
