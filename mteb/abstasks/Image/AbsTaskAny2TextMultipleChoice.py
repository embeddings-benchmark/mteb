from __future__ import annotations

import logging
from typing import Any

from datasets import Dataset

from ...encoder_interface import Encoder, EncoderWithQueryCorpusEncode
from ...evaluation.evaluators import Any2TextMultipleChoiceEvaluator
from ...load_results.mteb_results import ScoresDict
from ..AbsTask import AbsTask

logger = logging.getLogger(__name__)


class AbsTaskAny2TextMultipleChoice(AbsTask):
    """Abstract class for Any to Text Multiple Choice tasks,
    where the queries and be either text or image, or both.
    This task assess interleaved encoding of queries,
    the similarity computed between the queries and the candidate choices is ranked.

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset.
    """

    query_modalities: list[str] | str = ["image", "text"]
    query_column_names: dict = {"image": "image", "text": "question"}
    label_column_name: str = "answer"
    choices_column_name: str = "choices"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _add_main_score(self, scores) -> None:
        scores["main_score"] = scores[self.metadata.main_score]

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ):
        pass

    def _evaluate_subset(
        self,
        model: Encoder | EncoderWithQueryCorpusEncode,
        dataset: Dataset,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        for modality in self.query_modalities:
            if modality not in self.query_column_names:
                raise KeyError(
                    f"query column name of modality {modality} is not defined"
                )
        evaluator = Any2TextMultipleChoiceEvaluator(
            dataset,
            query_modalities=self.query_modalities,
            query_column_names=self.query_column_names,
            label_column_name=self.label_column_name,
            choices_column_name=self.choices_column_name,
            task_name=self.metadata.name,
            **kwargs,
        )
        scores = evaluator(model, encode_kwargs=encode_kwargs)
        self._add_main_score(scores)
        return scores
