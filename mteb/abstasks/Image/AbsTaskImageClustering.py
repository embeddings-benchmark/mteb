from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from mteb.abstasks import AbsTask
from mteb.encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from mteb.evaluation.evaluators import ImageClusteringEvaluator
from mteb.load_results.mteb_results import HFSubset, ScoresDict

logger = logging.getLogger(__name__)


class AbsTaskImageClustering(AbsTask):
    """Abstract class for Clustering tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        image: Image.Image
        label: int
    """

    image_column_name: str = "image"
    label_column_name: str = "label"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores: dict[HFSubset, ScoresDict]) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _evaluate_subset(
        self,
        model: EncoderWithQueryCorpusEncode | Encoder,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        evaluator = ImageClusteringEvaluator(
            dataset[self.image_column_name],
            dataset[self.label_column_name],
            task_name=self.metadata.name,
            **kwargs,
        )
        metrics = evaluator(model, encode_kwargs=encode_kwargs)
        scores = {"v_measure": metrics["v_measure"]}
        self._add_main_score(scores)
        return scores
