from __future__ import annotations

import itertools
import logging
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, label_ranking_average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MultiLabelBinarizer

from mteb.abstasks.TaskMetadata import HFSubset

from ...encoder_interface import Encoder
from ..AbsTask import AbsTask, ScoresDict

logger = logging.getLogger(__name__)



class AbsTaskAudioMultilabelClassification(AbsTask):
    """Abstract class for audio multioutput classification tasks
    The similarity is computed between pairs and the results are ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It must contain the following columns:
        audio: list[??]
        labels: list[Hashable]
    """

    audio_column_name: str = "audio"
    label_column_name: str = "labels"

    classifier = MultiOutputClassifier(estimator=LogisticRegression())

    def __init__(
        self,
        n_experiments=None,
        samples_per_label=None,
        batch_size=32,
        **kwargs,
    ):
        pass

    def _add_main_score(self, scores):
        pass

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        """To be implemented by the concrete task class"""
        pass

    def evaluate(
        self,
        model: Encoder,
        eval_split: str = "test",
        train_split: str = "train",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        pass

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset,
        eval_split: str = "test",
        train_split: str = "train",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> ScoresDict:
        pass

    def _undersample_data_indices(self, y, samples_per_label, idxs=None):
        """Undersample data to have samples_per_label samples of each label"""
        pass
