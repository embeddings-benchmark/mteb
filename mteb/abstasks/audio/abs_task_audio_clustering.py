import itertools
import logging
import random
from typing import Any

import numpy as np
from datasets import DatasetDict
from torch.utils.data import DataLoader

from mteb._create_dataloaders import _custom_collate_fn
from mteb.abstasks import AbsTask
from mteb.abstasks.clustering import _evaluate_clustering_bootstrapped
from mteb.abstasks.task_metadata import HFSubset
from mteb.models.models_protocols import EncoderProtocol
from mteb.types import ScoresDict

logger = logging.getLogger(__name__)


class AbsTaskAudioClustering(AbsTask):
    """Abstract class for AudioClustering tasks that is based on the AbsTaskClusteringFast class.
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata.eval_splits, and assign it to self.dataset. It must contain the following columns:
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

    def _calculate_descriptive_statistics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        pass

    def _evaluate_subset(
        self,
        model: EncoderProtocol,
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
            downsampled_dataset = dataset.select(example_indices)

        dataloader = DataLoader(
            downsampled_dataset,
            batch_size=encode_kwargs["batch_size"],
            collate_fn=_custom_collate_fn,
        )

        embeddings = model.encode(
            dataloader,
            batch_size=encode_kwargs["batch_size"],
            task_metadata=self.metadata,
            # todo: temporary fix, until full refactoring of task
            hf_subset="test",
            hf_split="test",
        )

        labels = []
        for label in downsampled_dataset[self.label_column_name]:
            if not isinstance(label, list):
                label = [label]
            labels.append(label)

        all_v_scores, all_assignments = _evaluate_clustering_bootstrapped(
            embeddings,
            labels,
            n_clusters=self.n_clusters,
            cluster_size=self.max_documents_per_cluster,
            kmean_batch_size=self.k_mean_batch_size,
            max_depth=self.max_depth,
            rng_state=rng_state,
            seed=self.seed,
        )
        v_measures = list(itertools.chain.from_iterable(all_v_scores.values()))
        mean_v_measure = np.mean(v_measures)
        v_std = np.std(v_measures)

        scores = {
            "v_measures": v_measures,
            "v_measure": float(mean_v_measure),
            "v_measure_std": v_std,
        }
        self._add_main_score(scores)
        return scores
