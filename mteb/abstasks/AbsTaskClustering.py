from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tqdm
from datasets import Dataset

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.load_results.mteb_results import ScoresDict

from ..evaluation.evaluators import ClusteringEvaluator
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskClustering(AbsTask):
    """Abstract class for Clustering tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sentences: list of str
        labels: list of str
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: EncoderWithQueryCorpusEncode | Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        v_measures = []
        for cluster_set in tqdm.tqdm(dataset, desc="Clustering"):
            evaluator = ClusteringEvaluator(
                cluster_set["sentences"],  # type: ignore
                cluster_set["labels"],  # type: ignore
                task_name=self.metadata.name,
                **kwargs,
            )
            metrics = evaluator(model, encode_kwargs=encode_kwargs)
            v_measures.append(metrics["v_measure"])

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        scores = {"v_measure": v_mean, "v_measure_std": v_std, "v_measures": v_measures}
        self._add_main_score(scores)
        return scores

    def process_split(self, split: str, lang: str | None = None) -> dict[str, float]:
        if lang:
            sentences = self.dataset[lang][split]["sentences"]
            labels = self.dataset[lang][split]["labels"]
        else:
            sentences = self.dataset[split]["sentences"]
            labels = self.dataset[split]["labels"]

        total_text_len = sum([len(t) for t in sentences])
        total_labels = []
        for label_list in labels:
            total_labels.extend(label_list)

        return {
            "num_texts": len(sentences),
            "num_labels": len(labels),
            "average_text_length": total_text_len / len(sentences),
            "average_label_count": len(total_labels) / len(labels),
            "unique_labels": len(set(total_labels)),
        }
