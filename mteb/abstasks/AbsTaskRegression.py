from __future__ import annotations

import logging
from typing import Any

from mteb.abstasks.TaskMetadata import DescriptiveStatistics, HFSubset
from mteb.encoder_interface import Encoder
from mteb.evaluation.evaluators.RegressionEvaluator import LinearRegressionEvaluator

from ..load_results.task_results import ScoresDict
from .AbsTask import AbsTask

logger = logging.getLogger(__name__)


class RegressionDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Regression

    Attributes:
      num_samples: number of samples in the dataset.
      number_of_characters: Total number of symbols in the dataset.
      num_texts_in_train: Number of texts in the train split

      min_text_length: Minimum length of text
      average_text_length: Average length of text
      max_text_length: Maximum length of text
      unique_text: Number of unique texts

      min_value: Minimum of the target variable
      average_value: Average of the target variable
      max_value: Maximum of the target variable
    """

    num_samples: int
    number_of_characters: int
    num_texts_in_train: int | None

    min_text_length: int
    average_text_length: float
    max_text_length: int
    unique_text: int

    min_value: float
    average_value: float
    max_value: float


class AbsTaskRegression(AbsTask):
    """Abstract class for regression tasks

    self.load_data() must generate a huggingface dataset with a split matching self.metadata_dict["eval_splits"], and assign it to self.dataset. It
    must contain the following columns:
        text: str
        value: float
    """

    def __init__(self, seed: int = 42, **kwargs: Any):
        super().__init__(seed, **kwargs)
        if hasattr(self, "metadata"):
            self.metadata

    def _evaluate_subset(
        self,
        model: Encoder,
        dataset,
        eval_split: str = "test",
        train_split: str = "train",
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        train_split = dataset[train_split]
        eval_split = dataset[eval_split]

        evaluator = LinearRegressionEvaluator(
            train_split["text"],
            train_split["value"],
            eval_split["text"],
            eval_split["value"],
            task_name=self.metadata.name,
            encode_kwargs=encode_kwargs,
            **kwargs,
        )
        scores = evaluator(model)
        return scores

    def _add_main_score(self, scores):
        scores["main_score"] = scores[self.metadata.main_score]

    def evaluate(
        self,
        model: Encoder,
        eval_split: str = "test",
        train_split: str = "train",
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        if not self.data_loaded:
            self.load_data()

        scores = {}
        hf_subsets = list(self.dataset) if self.is_multilingual else ["default"]

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

    def __hash__(self) -> int:
        return hash(self.metadata)

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> RegressionDescriptiveStatistics:
        train_text = []
        if hf_subset:
            texts = self.dataset[hf_subset][split]["text"]
            values = self.dataset[hf_subset][split]["value"]
            if split != "train":
                train_text = self.dataset[hf_subset]["train"]["text"]
        elif compute_overall:
            texts = []
            values = []
            for lang_subset in self.metadata.eval_langs:
                texts.extend(self.dataset[lang_subset][split]["text"])
                values.extend(self.dataset[lang_subset][split]["value"])
                if split != "train":
                    train_text.extend(self.dataset[lang_subset]["train"]["text"])
        else:
            texts = self.dataset[split]["text"]
            values = self.dataset[split]["value"]
            if split != "train":
                train_text = self.dataset["train"]["text"]

        text_lengths = [len(t) for t in texts]
        total_text_length = sum(text_lengths)

        num_texts_in_train_val = (
            len(set(texts) & set(train_text)) if split != "train" else None
        )

        return RegressionDescriptiveStatistics(
            num_samples=len(texts),
            number_of_characters=total_text_length,
            num_texts_in_train=num_texts_in_train_val,
            min_text_length=min(text_lengths) if text_lengths else 0,
            average_text_length=(total_text_length / len(texts))
            if len(texts) > 0
            else 0,
            max_text_length=max(text_lengths) if text_lengths else 0,
            unique_text=len(set(texts)),
            min_value=min(values) if values else 0.0,
            average_value=(sum(values) / len(values)) if len(values) > 0 else 0.0,
            max_value=max(values) if values else 0.0,
        )
