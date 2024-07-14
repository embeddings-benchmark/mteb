from __future__ import annotations

import logging
from typing import Any

import numpy as np
import tqdm
from datasets import Dataset

from mteb.abstasks import AbsTask
from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.evaluation.evaluators import ClusteringEvaluator
from mteb.load_results.mteb_results import HFSubset, ScoresDict

logger = logging.getLogger(__name__)


class AbsTaskImageClustering(AbsTask):
    """Abstract class for Clustering tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        sentences: list of str
        labels: list of str
    """

    image_column_name: str = "image"
    label_column_name: str = "label"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores: dict[HFSubset, ScoresDict]) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def evaluate(
        self,
        model,
        eval_split: str = "test",
        train_split: str = "train",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        scores = {}
        hf_subsets = [l for l in self.dataset] if self.is_multilingual else ["default"]

        for hf_subset in hf_subsets:
            logger.info(
                f"\nTask: {self.metadata.name}, split: {eval_split}, subset: {hf_subset}. Running..."
            )

            if hf_subset not in self.dataset and hf_subset == "default":
                ds = self.dataset
            else:
                ds = self.dataset[hf_subset]
            scores[hf_subset] = self._evaluate_subset(
                model,
                ds,
                eval_split,
                train_split,
                encode_kwargs=encode_kwargs,
                **kwargs,
            )
            self._add_main_score(scores[hf_subset])

        return scores

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
                cluster_set[self.image_column_name],
                cluster_set[self.label_column_name],
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
