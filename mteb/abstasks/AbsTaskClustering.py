from __future__ import annotations

import logging

import numpy as np
import tqdm
from datasets import Dataset

from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.MTEBResults import ScoresDict

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
        self, model: EncoderWithQueryCorpusEncode | Encoder, dataset: Dataset, **kwargs
    ) -> ScoresDict:
        v_measures = []
        for cluster_set in tqdm.tqdm(dataset, desc="Clustering"):
            evaluator = ClusteringEvaluator(
                cluster_set["sentences"],  # type: ignore
                cluster_set["labels"],  # type: ignore
                **kwargs,
            )
            metrics = evaluator(model)
            v_measures.append(metrics["v_measure"])

        v_mean = np.mean(v_measures)
        v_std = np.std(v_measures)
        scores = {"v_measure": v_mean, "v_measure_std": v_std, "v_measures": v_measures}
        self._add_main_score(scores)
        return scores
