import logging
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import (
    TaskDomain,
    TaskType,
)
from mteb.types import (
    ISOLanguage,
    ISOLanguageScript,
    Modalities,
    Score,
    ScoresDict,
    SplitName,
)

from .task_result import TaskResult

logger = logging.getLogger(__name__)


def _aggregate_and_pivot(
    df: pd.DataFrame,
    columns: list[str],
    aggregation_level: Literal["subset", "split", "task"],
    format: Literal["wide", "long"],
    aggregation_fn: Callable[[list[Score]], Any] | None,
) -> pd.DataFrame:
    if aggregation_level == "subset":
        index_columns = ["task_name", "split", "subset"]

    elif aggregation_level == "split":
        index_columns = ["task_name", "split"]

    elif aggregation_level == "task":
        index_columns = ["task_name"]

    # perform aggregation
    if aggregation_fn is None:
        aggregation_fn = np.mean

    if format == "wide":
        return df.pivot_table(
            index=index_columns,
            columns=columns,
            values="score",
            aggfunc=aggregation_fn,
        ).reset_index()
    elif format == "long":
        return (
            df.groupby(columns + index_columns)
            .agg(score=("score", aggregation_fn))
            .reset_index()
        )


class ModelResult(BaseModel):
    """Data class to hold the results of a model on a set of tasks.

    Attributes:
        model_name: Name of the model.
        model_revision: Revision of the model.
        task_results: List of TaskResult objects.
    """

    model_name: str
    model_revision: str | None
    task_results: list[TaskResult]
    default_modalities: list[Modalities] = Field(
        default_factory=lambda: ["text"], alias="modalities"
    )
    model_config = (
        ConfigDict(  # to free up the name model_* which is otherwise protected
            protected_namespaces=(),
        )
    )

    def __repr__(self) -> str:
        n_entries = len(self.task_results)
        return f"ModelResult(model_name={self.model_name}, model_revision={self.model_revision}, task_results=[...](#{n_entries}))"

    @classmethod
    def from_validated(cls, **data: dict[str, Any]) -> Self:
        """Create a ModelResult from validated data.

        Args:
            data: The validated data.
        """
        data["task_results"] = [
            TaskResult.from_validated(**res) for res in data["task_results"]
        ]
        return cls.model_construct(**data)

    def _filter_tasks(
        self,
        task_names: list[str] | None = None,
        languages: list[str] | None = None,
        domains: list[TaskDomain] | None = None,
        task_types: list[TaskType] | None = None,
        modalities: list[Modalities] | None = None,
        is_public: bool | None = None,
    ) -> Self:
        new_task_results = []
        for task_result in self.task_results:
            if (task_names is not None) and (task_result.task_name not in task_names):
                continue
            if languages is not None:
                task_languages = task_result.languages
                if not any(lang in task_languages for lang in languages):
                    continue
            if domains is not None:
                task_domains = task_result.domains
                if not any(domain in task_domains for domain in domains):
                    continue
            if (task_types is not None) and (task_result.task_type not in task_types):
                continue
            if modalities is not None:
                task_modalities = getattr(task_result, "modalities", [])
                if not any(modality in task_modalities for modality in modalities):
                    continue
            if (is_public is not None) and (task_result.is_public is not is_public):
                continue
            new_task_results.append(task_result)
        return type(self).model_construct(
            model_name=self.model_name,
            model_revision=self.model_revision,
            task_results=new_task_results,
        )

    def select_tasks(self, tasks: Sequence[AbsTask]) -> Self:
        """Select tasks from the ModelResult based on a list of AbsTask objects.

        Args:
            tasks: A sequence of AbsTask objects to select from the ModelResult.
        """
        task_name_to_task = {task.metadata.name: task for task in tasks}
        new_task_results = [
            task_res.validate_and_filter_scores(task_name_to_task[task_res.task_name])
            for task_res in self.task_results
            if task_res.task_name in task_name_to_task
        ]
        return type(self).model_construct(
            model_name=self.model_name,
            model_revision=self.model_revision,
            task_results=new_task_results,
        )

    def _get_scores(
        self,
        splits: list[SplitName] | None = None,
        languages: list[ISOLanguage | ISOLanguageScript] | None = None,
        scripts: list[ISOLanguageScript] | None = None,
        getter: Callable[[ScoresDict], Score] | None = None,
        aggregation: Callable[[list[Score]], Any] | None = None,
        format: Literal["wide", "long"] = "wide",
    ) -> dict | list:
        if (getter is not None) or (aggregation is not None) or (scripts is not None):
            use_fast = False
            getter = (
                getter if getter is not None else lambda scores: scores["main_score"]
            )
            aggregation = aggregation if aggregation is not None else np.mean
        else:
            use_fast = True
        if format == "wide":
            scores = {}
            for res in self.task_results:
                try:
                    if use_fast:
                        scores[res.task_name] = res._get_score_fast(
                            splits=splits,  # type: ignore
                            languages=languages,  # type: ignore
                        )
                    else:
                        scores[res.task_name] = res.get_score(
                            splits=splits,
                            languages=languages,
                            aggregation=aggregation,  # type: ignore
                            getter=getter,  # type: ignore
                            scripts=scripts,
                        )
                except Exception as e:
                    warnings.warn(
                        f"Couldn't get scores for {res.task_name} due to {e}."
                    )
            return scores
        if format == "long":
            entries = []
            for task_res in self.task_results:
                try:
                    if use_fast:
                        score = task_res._get_score_fast(
                            splits=splits,
                            languages=languages,  # type: ignore
                        )
                    else:
                        score = task_res.get_score(
                            splits=splits,
                            languages=languages,
                            aggregation=aggregation,  # type: ignore
                            getter=getter,  # type: ignore
                            scripts=scripts,
                        )
                    entry = dict(
                        model_name=self.model_name,
                        model_revision=self.model_revision,
                        task_name=task_res.task_name,
                        score=score,
                        mteb_version=task_res.mteb_version,
                        dataset_revision=task_res.dataset_revision,
                        evaluation_time=task_res.evaluation_time,
                        kg_co2_emissions=task_res.kg_co2_emissions,
                    )
                    entries.append(entry)
                except Exception as e:
                    warnings.warn(
                        f"Couldn't get scores for {task_res.task_name} due to {e}."
                    )
            return entries

    def _get_score_for_table(self) -> list[dict[str, str | float]]:
        scores_data = []
        model_name = self.model_name
        for task_result in self.task_results:
            task_name = task_result.task_name
            for split, scores_list in task_result.scores.items():
                for score_item in scores_list:
                    row = {
                        "model_name": model_name,
                        "model_revision": self.model_revision,
                        "task_name": task_name,
                        "split": split,
                        "subset": score_item.get("hf_subset", "default"),
                        "score": score_item.get("main_score", None),
                    }

                    scores_data.append(row)

        return scores_data

    def to_dataframe(
        self,
        aggregation_level: Literal["subset", "split", "task"] = "task",
        aggregation_fn: Callable[[list[Score]], Any] | None = None,
        include_model_revision: bool = False,
        format: Literal["wide", "long"] = "wide",
    ) -> pd.DataFrame:
        """Get a DataFrame with the scores for all models and tasks.

        The DataFrame will have the following columns in addition to the metadata columns:

        - model_name: The name of the model.
        - task_name: The name of the task.
        - score: The main score of the model on the task.

        In addition, the DataFrame can have the following columns depending on the aggregation level:

        - split: The split of the task. E.g. "test", "train", "validation".
        - subset: The subset of the task. E.g. "en", "fr-en".

        Afterwards, the DataFrame will be aggregated according to the aggregation method and pivoted to either a wide format.

        Args:
            aggregation_level: The aggregation to use. Can be one of:
                - "subset"/None: No aggregation will be done. The DataFrame will have one row per model, task, split and subset.
                - "split": Aggregates the scores by split. The DataFrame will have one row per model, task and split.
                - "task": Aggregates the scores by task. The DataFrame will have one row per model and task.
            aggregation_fn: The function to use for aggregation. If None, the mean will be used.
            include_model_revision: If True, the model revision will be included in the DataFrame. If False, it will be excluded.
            format: The format of the DataFrame. Can be one of:
                - "wide": The DataFrame will be of shape (number of tasks, number of models). Scores will be in the cells.
                - "long": The DataFrame will of length (number of tasks * number of model). Scores will be in columns.

        Returns:
            A DataFrame with the scores for all models and tasks.
        """
        scores_data = self._get_score_for_table()

        if not scores_data:
            logger.warning("No scores data available. Returning empty DataFrame.")
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(scores_data)

        _columns = ["model_name"]
        if include_model_revision is False:
            df = df.drop(columns=["model_revision"])
        else:
            _columns.append("model_revision")

        return _aggregate_and_pivot(
            df,
            columns=_columns,
            aggregation_level=aggregation_level,
            format=format,
            aggregation_fn=aggregation_fn,
        )

    def __hash__(self) -> int:
        return id(self)

    def __iter__(self) -> Iterable[TaskResult]:
        return iter(self.task_results)

    def __getitem__(self, index) -> TaskResult:
        return self.task_results[index]

    def __len__(self) -> int:
        return len(self.task_results)

    @property
    def languages(self) -> list[str]:
        """Get all languages in the model results.

        Returns:
            A list of languages in the model results.
        """
        langs = []
        for task_res in self.task_results:
            langs.extend(task_res.languages)
        return list(set(langs))

    @property
    def domains(self) -> list[str]:
        """Get all domains in the model results.

        Returns:
            A list of domains in the model results.

        """
        ds = []
        for task_res in self.task_results:
            ds.extend(task_res.domains)
        return list(set(ds))

    @property
    def task_types(self) -> list[str]:
        """Get all task types in the model results.

        Returns:
            A list of task types in the model results.
        """
        return list({task_res.task_type for task_res in self.task_results})

    @property
    def task_names(self) -> list[str]:
        """Get all task names in the model results.

        Returns:
            A list of task names in the model results.
        """
        return [task_res.task_name for task_res in self.task_results]

    @property
    def modalities(self) -> list[str]:
        """Get all modalities in the task results.

        Returns:
            A list of modalities in the task results.
        """
        mods = []
        for task_res in self.task_results:
            task_modalities = getattr(task_res, "modalities", [])
            mods.extend(task_modalities)
        if not mods:
            mods = self.default_modalities
        return list(set(mods))
