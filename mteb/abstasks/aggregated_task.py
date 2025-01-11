from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from pydantic import field_validator

from mteb.abstasks.TaskMetadata import DescriptiveStatistics, HFSubset, TaskMetadata
from mteb.encoder_interface import Encoder
from mteb.load_results.task_results import TaskResult

from .AbsTask import AbsTask, ScoresDict

logger = logging.getLogger(__name__)


class AggregatedTaskMetadata(TaskMetadata):
    """A derivative of the taskmetadata used for aggregated of tasks. Can e.g. be used to create custom tasks
    which are a combination of existing task. For an example see CQADupstackRetrieval.

    The attributes are the same as TaskMetadata, with a few exceptions described below.

    Attributes:
        dataset: Always None as the task dataset is specified in its subtasks
        prompt: Always None as the task prompt is specified in its subtasks
        tasks: A list of tasks
    """

    dataset: None = None
    prompt: None = None
    tasks: list[AbsTask]

    @field_validator("dataset")
    def _check_dataset_path_is_specified(
        cls, dataset: dict[str, Any]
    ) -> dict[str, Any]:
        return dataset  # skip validation

    @field_validator("dataset")
    def _check_dataset_revision_is_specified(
        cls, dataset: dict[str, Any]
    ) -> dict[str, Any]:
        return dataset  # skip validation

    @field_validator("prompt")
    def _check_prompt_is_valid(cls, prompt: None) -> None:
        return prompt  # skip validation


class AbsTaskAggregated(AbsTask):
    metadata: AggregatedTaskMetadata
    abstask_prompt: None = None

    def __init__(self, seed: int = 42, **kwargs: Any):
        self.tasks = self.metadata.tasks
        self.save_suffix = kwargs.get("save_suffix", "")

        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

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

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: Dataset,
        *,
        parallel: bool = False,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> ScoresDict:
        raise NotImplementedError()

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> DescriptiveStatistics:
        # it is a bit annoying that we have remove
        # functionality from a class. Let me know if you have a better way to doing this.
        raise NotImplementedError()
