from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from mteb.abstasks import AbsTask

if TYPE_CHECKING:
    from mteb.abstasks.TaskMetadata import HFSubset
    from mteb.encoder_interface import Encoder
    from mteb.load_results.task_results import TaskResult

    from .AbsTask import ScoresDict

logger = logging.getLogger(__name__)


class AggregateTaskMetadata(BaseModel):
    """Metadata for an aggregation of tasks.

    Attributes:
        name: The name of the task.
        description: A description of the task. Should explain the aggregation.
        reference: A URL to the documentation of the task. E.g. a published paper.
        tasks: A list of tasks, the majority of the metadata is described within its tasks.
                main_score: The main score used for evaluation.
       main_score: the main score of the task
        eval_splits: The splits of the dataset used for evaluation.
        bibtex_citation: The BibTeX citation for the dataset. Should be an empty string if no citation is available.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    reference: str
    tasks: list[AbsTask]
    main_score: str
    bibtex_citation: str
    type: Literal["aggregate-task"] = "aggregate-task"


class AggregateTask:
    metadata: AggregateTaskMetadata
    hf_subsets: list[str] = [
        "default"
    ]  # since there is no subset we use the "default" naming scheme

    def __init__(self, **kwargs: Any):
        self.tasks = self.metadata.tasks

    def task_results_to_score(self, task_results: list[TaskResult]) -> ScoresDict:
        """The function that aggregated scores"""
        main_scores = []
        for task_res in task_results:
            main_scores.append(
                task_res.get_score(
                    getter=lambda scores: scores[self.metadata.main_score]
                )
            )
        return {self.metadata.main_score: np.mean(main_scores)}

    def load_data(self, **kwargs: Any) -> None:
        for task in self.tasks:
            task.load_data()

        self.data_loaded = True
