from __future__ import annotations

import logging
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from mteb.abstasks.TaskMetadata import HFSubset
from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import TaskResult

from .AbsTask import AbsTask, ScoresDict

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
    eval_splits: list[str] = ["test"]
    bibtex_citation: str


class AggregateTask:
    metadata: AggregateTaskMetadata

    def __init__(self, **kwargs: Any):
        self.tasks = self.metadata.tasks

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        mteb_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        from mteb.evaluation.MTEB import MTEB  # to prevent circular imports

        if subsets_to_run:
            logger.warning(
                "Specifying which subset to run is not supported for aggregated tasks. It will be ignored."
            )

        bench = MTEB(tasks=self.tasks)
        task_results = bench.run(
            model=model,
            encode_kwargs=encode_kwargs,
            eval_subsets=None,
            eval_splits=[split],
            verbosity=0,
            **mteb_kwargs,
        )
        return {"default": self.task_results_to_score(task_results)}

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
