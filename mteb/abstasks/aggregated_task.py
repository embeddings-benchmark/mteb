from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from mteb.abstasks.AbsTask import AbsTask
from mteb.abstasks.aggregate_task_metadata import AggregateTaskMetadata

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

    from mteb.abstasks.TaskMetadata import DescriptiveStatistics, HFSubset
    from mteb.encoder_interface import Encoder
    from mteb.load_results.task_results import TaskResult

    from .AbsTask import ScoresDict

logger = logging.getLogger(__name__)


class AbsTaskAggregate(AbsTask):
    metadata: AggregateTaskMetadata
    superseded_by: None | str = None
    hf_subset = "default"  # since there is no subset we use the "default" naming scheme
    _eval_splits: list[str] | None = None

    def __init__(self, **kwargs: Any):
        self.tasks = self.metadata.tasks
        self.taskname_to_task = {task.metadata.name: task for task in self.tasks}

    def task_results_to_scores(
        self, task_results: list[TaskResult]
    ) -> dict[str, dict[HFSubset, ScoresDict]]:
        """The function that aggregated scores. Can be redefined to allow for custom aggregations."""
        scores = {}
        subsets = (
            self.metadata.eval_langs.keys()
            if isinstance(self.metadata.eval_langs, dict)
            else None
        )
        eval_langs = (
            self.metadata.eval_langs.values()
            if isinstance(self.metadata.eval_langs, dict)
            else [self.metadata.eval_langs]
        )
        for split in self.metadata.eval_splits:
            main_scores = []
            for task_res in task_results:
                for langs in eval_langs:
                    main_scores.append(
                        task_res.get_score_fast(
                            languages=[lang.split("-")[0] for lang in langs],
                            splits=self.metadata.eval_splits,
                            subsets=subsets,
                        )
                    )
            main_score = np.mean(main_scores)
            scores[split] = {
                "default": {
                    self.metadata.main_score: main_score,
                    "main_score": main_score,
                }
            }
        return scores

    def combine_task_results(self, task_results: list[TaskResult]) -> TaskResult:
        """Combined the task results for using `task_results_to_scores`. Do not redefine this function if you want to implement a custom aggregation.
        Instead redefin `task_results_to_scores`.
        """
        from mteb.load_results.task_results import (
            TaskResult,  # to prevent circular imports, # TODO: can potentially likely be out of function in in v2.0.0
        )

        eval_times = [tr.evaluation_time for tr in task_results if tr.evaluation_time]
        if len(eval_times) != len(task_results):
            logger.info(
                f"Loaded results does not include runtime. Therefore evaluation of {self.metadata.name} "
                + "can't be computed. Setting it to None."
            )
            eval_time = np.nan
        else:
            eval_time = sum(eval_times)

        kg_co2_emissions_ = [
            tr.kg_co2_emissions for tr in task_results if tr.kg_co2_emissions
        ]
        if len(kg_co2_emissions_) != len(task_results):
            logger.info(
                f"Loaded results does not include co2-eq emissions. Therefore evaluation of {self.metadata.name} "
                + "can't be computed. Setting it to None."
            )
            kg_co2_emissions = np.nan
        else:
            kg_co2_emissions = sum(kg_co2_emissions_)

        task_res = TaskResult.from_task_results(
            self,
            scores=self.task_results_to_scores(task_results),
            evaluation_time=eval_time,
            kg_co2_emissions=kg_co2_emissions,
        )
        mteb_versions = {tr.mteb_version for tr in task_results}
        if len(mteb_versions) != 1:
            logger.warning(
                f"All tasks of {self.metadata.name} is not run using the same version."
            )
            task_res.mteb_version = None
        task_res.mteb_version = task_results[0].mteb_version
        return task_res

    def check_if_dataset_is_superseded(self):
        """Check if the dataset is superseded by a newer version"""
        if self.superseded_by:
            logger.warning(
                f"Dataset '{self.metadata.name}' is superseded by '{self.superseded_by}', you might consider using the newer version of the dataset."
            )

    def filter_eval_splits(self, eval_splits: list[str] | None) -> AbsTaskAggregate:
        """Filter the evaluation splits of the task."""
        self._eval_splits = eval_splits
        return self

    def evaluate(
        self,
        model: Encoder,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs: Any,
    ) -> dict[HFSubset, ScoresDict]:
        # TODO: If we refactor the runner to at least have a subfunction mteb.run_task(model, task) we could use that here
        raise NotImplementedError(
            "Aggregate tasks can't be evaluated directly. Instead run it using the MTEB class."
        )

    def _evaluate_subset(
        self,
        model: Encoder,
        data_split: DatasetDict | Dataset,
        encode_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> ScoresDict:
        raise NotImplementedError(
            "Aggregate tasks does not implement a _evaluate_subset. Instead use the individual tasks."
        )

    def _calculate_metrics_from_split(
        self, split: str, hf_subset: str | None = None, compute_overall: bool = False
    ) -> DescriptiveStatistics:
        raise NotImplementedError(
            "Aggregate tasks does not implement a _calculate_metrics_from_split. Instead use the individual tasks."
        )

    @property
    def is_aggregate(self):  # Overrides the is_aggregate method on AbsTask
        return True

    @property
    def eval_splits(self) -> list[str]:
        if self._eval_splits:
            return self._eval_splits
        return self.metadata.eval_splits
