from __future__ import annotations

import itertools
import logging
import random
from typing import Any

import numpy as np
from datasets import DatasetDict

from mteb.abstasks.TaskMetadata import HFSubset

from ...encoder_interface import Encoder
from ..AbsTask import AbsTask, ScoresDict
from ..AbsTaskClusteringFast import evaluate_clustering_bootstrapped

logger = logging.getLogger(__name__)


class AbsTaskAudioClustering(AbsTask):
    """Abstract class for AudioClustering tasks that is based on the AbsTaskClusteringFast class.
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        audio: datasets.Audio
        label: int
    """

    max_fraction_of_documents_to_embed: float | None = 0.04
    max_document_to_embed: int | None = None
    max_documents_per_cluster: int = 16_384
    n_clusters: int = 10
    k_mean_batch_size: int = 512
    max_depth = None
    audio_column_name: str = "audio"
    label_column_name: str = "labels"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores: dict[HFSubset, ScoresDict]) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        pass

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset: DatasetDict,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[str, float | dict[str, list[float]]]:
        rng_state = random.Random(self.seed)

        if (
            self.max_document_to_embed is not None
            and self.max_fraction_of_documents_to_embed is not None
        ):
            raise Exception(
                "Both max_document_to_embed and max_fraction_of_documents_to_embed are set. Please only set one."
            )

        if (
            self.max_document_to_embed is None
            and self.max_fraction_of_documents_to_embed is None
        ):
            downsampled_dataset = dataset
        else:
            if self.max_fraction_of_documents_to_embed is not None:
                max_documents_to_embed = int(
                    self.max_fraction_of_documents_to_embed * len(dataset)
                )
            else:
                max_documents_to_embed = self.max_document_to_embed

            max_documents_to_embed = min(len(dataset), max_documents_to_embed)  # type: ignore
            example_indices = rng_state.sample(
                range(len(dataset)), k=max_documents_to_embed
            )
            downsampled_dataset = dataset.select(example_indices)  # type: ignore

        # Filter out empty audio samples before embedding to maintain alignment with labels
        valid_indices = []
        valid_audio = []
        valid_labels = []
        
        for i, audio_sample in enumerate(downsampled_dataset[self.audio_column_name]):
            is_valid = False
            
            # Check if audio sample is empty
            if isinstance(audio_sample, dict) and "array" in audio_sample:
                audio_array = audio_sample["array"]
                # More thorough empty check
                if (hasattr(audio_array, '__len__') and 
                    len(audio_array) > 0 and 
                    not (hasattr(audio_array, 'sum') and audio_array.sum() == 0 and len(audio_array) < 100)):
                    is_valid = True
                else:
                    logger.warning(f"Skipping empty/invalid audio sample at index {i}: shape={getattr(audio_array, 'shape', 'unknown')}")
            elif isinstance(audio_sample, (list, tuple, np.ndarray)) and len(audio_sample) > 0:
                is_valid = True
            elif isinstance(audio_sample, str):  # File path
                # For file paths, we'll assume valid here and let model handle loading errors
                # In a production system, you might want to pre-validate file existence
                is_valid = True
            else:
                logger.warning(f"Skipping unknown audio format at index {i}: type={type(audio_sample)}")
            
            if is_valid:
                valid_indices.append(i)
                valid_audio.append(audio_sample)
                label = downsampled_dataset[self.label_column_name][i]
                if not isinstance(label, list):
                    label = [label]
                valid_labels.append(label)
        
        if not valid_audio:
            logger.error("No valid audio samples found in dataset")
            return {"v_measure": 0.0, "v_measure_std": 0.0, "v_measures": {}}
        
        logger.info(f"Processing {len(valid_audio)} valid audio samples out of {len(downsampled_dataset)}")
        
        # Log label distribution for debugging
        label_counts = {}
        for label_list in valid_labels:
            label = label_list[0] if isinstance(label_list, list) else label_list
            label_counts[label] = label_counts.get(label, 0) + 1
        
        logger.info(f"Label distribution after filtering: {label_counts}")
        
        if "batch_size" not in encode_kwargs:
            encode_kwargs["batch_size"] = 32
        embeddings = model.get_audio_embeddings(
            valid_audio,
            batch_size=encode_kwargs["batch_size"],
        )

        labels = valid_labels

        all_v_scores = evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=self.n_clusters,
            cluster_size=self.max_documents_per_cluster,
            kmean_batch_size=self.k_mean_batch_size,
            max_depth=self.max_depth,
            rng_state=rng_state,
        )
        v_measures = list(itertools.chain.from_iterable(all_v_scores.values()))
        mean_v_measure = np.mean(v_measures)
        v_std = np.std(v_measures)

        scores = {
            "v_measures": all_v_scores,
            "v_measure": float(mean_v_measure),
            "v_measure_std": v_std,
        }
        self._add_main_score(scores)
        return scores
