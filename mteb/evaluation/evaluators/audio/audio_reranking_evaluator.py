import logging
from typing import Any

import numpy as np
import torch
import tqdm
from datasets.features._torchcodec import AudioDecoder
from sklearn.metrics import average_precision_score

from mteb._create_dataloaders import _create_audio_dataloader_from_audio_list
from mteb._evaluators import Evaluator
from mteb._evaluators.retrieval_metrics import confidence_scores, nauc
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.models_protocols import EncoderProtocol
from mteb.similarity_functions import cos_sim
from mteb.types import PromptType

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    return batch


class AudioRerankingEvaluator(Evaluator):
    """This class evaluates an EncoderProtocol model for the task of audio re-ranking.
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
        task_metadata: TaskMetadata | None = None,
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
        self.task_metadata = task_metadata
        self.k_values = k_values
        self.encode_kwargs = encode_kwargs
        self.transform = transform

        if "batch_size" not in self.encode_kwargs:
            self.encode_kwargs["batch_size"] = 32  # Smaller batch size for audio

        ### Remove samples with empty positive / negative set
        filtered_samples = []
        for sample in self.samples:
            if (
                isinstance(sample[self.positive_column_name], AudioDecoder)
                and isinstance(sample[self.negative_column_name], AudioDecoder)
            ) or (
                len(sample[self.positive_column_name]) > 0
                and len(sample[self.negative_column_name]) > 0
            ):
                filtered_samples.append(sample)

        self.samples = filtered_samples

        if len(self.samples) == 0:
            raise ValueError(
                "No valid samples found with non-empty positive and negative sets"
            )

    def __call__(self, model: EncoderProtocol):
        scores = self.compute_metrics(model)
        return scores

    def compute_metrics(self, model: EncoderProtocol):
        return (
            self.compute_metrics_batched(model)
            if self.use_batched_encoding
            else self.compute_metrics_individual(model)
        )

    def compute_metrics_batched(self, model: EncoderProtocol):
        """Computes the metrics in a batched way, by batching all queries and
        all documents together
        """
        logger.info("Encoding queries...")
        all_query_audios = [sample[self.query_column_name] for sample in self.samples]
        query_dataloader = _create_audio_dataloader_from_audio_list(all_query_audios)

        all_query_embs = np.asarray(
            model.encode(
                query_dataloader,
                task_metadata=self.task_metadata,
                prompt_type=PromptType.query,
                hf_subset="test",
                hf_split="test",
                **self.encode_kwargs,
            )
        )

        all_mrr_scores = []
        all_ap_scores = []
        all_conf_scores = []

        # Collect all documents
        all_docs = []
        for sample in self.samples:
            all_docs.append(sample[self.positive_column_name])
            all_docs.append(sample[self.negative_column_name])

        all_docs_embs = np.asarray(
            model.encode(
                _create_audio_dataloader_from_audio_list(all_docs),
                task_metadata=self.task_metadata,
                prompt_type=PromptType.document,
                hf_subset="test",
                hf_split="test",
                **self.encode_kwargs,
            )
        )

        # Compute scores
        logger.info("Evaluating...")
        docs_idx = 0
        for i, instance in enumerate(self.samples):
            query_emb = all_query_embs[i : i + 1]  # Single query embedding

            num_pos = 1
            num_neg = 1
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

    def compute_metrics_individual(self, model: EncoderProtocol):
        """Encodes and evaluates each (query, positive, negative) tuple individually.
        Slower but more memory efficient.
        """
        all_mrr_scores = []
        all_ap_scores = []
        all_conf_scores = []

        for instance in tqdm.tqdm(self.samples, desc="Evaluating samples"):
            query_audio = instance[self.query_column_name]
            positive_audios = list(instance[self.positive_column_name])
            negative_audios = list(instance[self.negative_column_name])

            if len(positive_audios) == 0 or len(negative_audios) == 0:
                continue

            docs_audios = positive_audios + negative_audios
            is_relevant = [True] * len(positive_audios) + [False] * len(negative_audios)

            # Encode query and documents
            query_emb = np.asarray(
                model.encode(
                    _create_audio_dataloader_from_audio_list([query_audio]),
                    task_metadata=self.task_metadata,
                    prompt_type=PromptType.query,
                    hf_subset="test",
                    hf_split="test",
                    **self.encode_kwargs,
                )
            )

            docs_emb = np.asarray(
                model.encode(
                    _create_audio_dataloader_from_audio_list(docs_audios),
                    task_metadata=self.task_metadata,
                    prompt_type=PromptType.document,
                    hf_subset="test",
                    hf_split="test",
                    **self.encode_kwargs,
                )
            )

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
                naucs_map = self.nauc_scores(all_conf_scores, all_ap_scores, "map")
                naucs_mrr = self.nauc_scores(all_conf_scores, all_mrr_scores, "mrr")
                results.update(naucs_map)
                results.update(naucs_mrr)
            except Exception as e:
                logger.warning(f"Error calculating nAUC: {e}")

        return results

    @staticmethod
    def nauc_scores(
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
            f"nAUC_{metric_name}_{fct}": nauc(all_conf_scores[fct], metrics)
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
