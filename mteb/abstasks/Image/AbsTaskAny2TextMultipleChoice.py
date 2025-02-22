from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from datasets import Dataset

from ...encoder_interface import Encoder
from ...evaluation.evaluators import Any2TextMultipleChoiceEvaluator
from ..AbsTask import AbsTask, ScoresDict
from ..TaskMetadata import DescriptiveStatistics

logger = logging.getLogger(__name__)


class Any2TextMutipleChoiceDescriptiveStatistics(DescriptiveStatistics):
    """Descriptive statistics for Any2TextMutipleChoice

    Attributes:
        num_samples: number of samples in the dataset.

        min_image_width: Minimum width of images
        average_image_width: Average width of images
        max_image_width: Maximum width of images

        min_image_height: Minimum height of images
        average_image_height: Average height of images
        max_image_height: Maximum height of images

        min_num_choices: Minimum number of choices
        average_num_choices: Average number of choices
        max_num_choices: Maximum number of choices

        answers: dict of answer frequencies

        min_question_length: Minimum length of questions
        average_question_length: Average length of questions
        max_question_length: Maximum length of questions
    """

    num_samples: int

    min_image_width: float
    average_image_width: float
    max_image_width: float

    min_image_height: float
    average_image_height: float
    max_image_height: float

    min_num_choices: int
    average_num_choices: float
    max_num_choices: int

    answers: dict[str, dict[str, int]]

    min_question_length: int
    average_question_length: float
    max_question_length: int


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
    ) -> Any2TextMutipleChoiceDescriptiveStatistics:
        imgs = self.dataset[split][self.query_column_names["image"]]
        questions = self.dataset[split][self.query_column_names["text"]]
        choices = self.dataset[split][self.choices_column_name]
        answers = self.dataset[split][self.label_column_name]

        num_samples = len(answers)
        answer_count = Counter(answers)
        img_widths, img_heights = [], []
        for img in imgs:
            width, height = img.size
            img_heights.append(height)
            img_widths.append(width)

        choices_len = [len(c) for c in choices]
        questions_len = [len(q) for q in questions]

        return Any2TextMutipleChoiceDescriptiveStatistics(
            num_samples=num_samples,
            min_image_width=min(img_widths),
            average_image_width=sum(img_widths) / len(img_widths),
            max_image_width=max(img_widths),
            min_image_height=min(img_heights),
            average_image_height=sum(img_heights) / len(img_heights),
            max_image_height=max(img_heights),
            min_num_choices=min(choices_len),
            average_num_choices=sum(choices_len) / len(choices_len),
            max_num_choices=max(choices_len),
            min_question_length=min(questions_len),
            average_question_length=sum(questions_len) / len(questions_len),
            max_question_length=max(questions_len),
            answers={
                str(answer): {"count": count} for answer, count in answer_count.items()
            },
        )

    def _evaluate_subset(
        self,
        model: Encoder,
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
