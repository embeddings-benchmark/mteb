from __future__ import annotations

import json
import logging
import warnings
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict, Field

from mteb.abstasks.AbsTask import AbsTask, ScoresDict
from mteb.abstasks.TaskMetadata import (
    ISO_LANGUAGE_SCRIPT,
    TASK_DOMAIN,
    TASK_TYPE,
)
from mteb.custom_validators import MODALITIES
from mteb.languages import ISO_LANGUAGE
from mteb.load_results.task_results import TaskResult
from mteb.models.overview import ModelMeta, get_model_metas

Split = str
Score = Any


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

    # TODO: v2, move to its own file model_result.py

    model_name: str
    model_revision: str | None
    task_results: list[TaskResult]
    default_modalities: list[MODALITIES] = Field(
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
    def from_validated(cls, **data) -> ModelResult:
        data["task_results"] = [
            TaskResult.from_validated(**res) for res in data["task_results"]
        ]
        return cls.model_construct(**data)

    def filter_tasks(
        self,
        task_names: list[str] | None = None,
        languages: list[str] | None = None,
        domains: list[TASK_DOMAIN] | None = None,
        task_types: list[TASK_TYPE] | None = None,
        modalities: list[MODALITIES] | None = None,
    ) -> ModelResult:
        # TODO: v2 see filter_tasks in BenchmarkResults - but can be moved to a private function or removed
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
            new_task_results.append(task_result)
        return type(self).model_construct(
            model_name=self.model_name,
            model_revision=self.model_revision,
            task_results=new_task_results,
        )

    def select_tasks(self, tasks: Sequence[AbsTask]) -> ModelResult:
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

    def get_scores(
        self,
        splits: list[Split] | None = None,
        languages: list[ISO_LANGUAGE | ISO_LANGUAGE_SCRIPT] | None = None,
        scripts: list[ISO_LANGUAGE_SCRIPT] | None = None,
        getter: Callable[[ScoresDict], Score] | None = None,
        aggregation: Callable[[list[Score]], Any] | None = None,
        format: Literal["wide", "long"] = "wide",
    ) -> dict | list:
        # TODO: Convert to private function in v2 - potentially remove
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
                        scores[res.task_name] = res.get_score_fast(
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
                        score = task_res.get_score_fast(
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
                    entry = dict(  # noqa
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


class BenchmarkResults(BaseModel):
    """Data class to hold the benchmark results of a model.

    Attributes:
        model_results: List of ModelResult objects.
    """

    model_results: list[ModelResult]
    model_config = (
        ConfigDict(  # to free up the name model_results which is otherwise protected
            protected_namespaces=(),
        )
    )

    def __repr__(self) -> str:
        n_models = len(self.model_results)
        return f"BenchmarkResults(model_results=[...](#{n_models}))"

    def __hash__(self) -> int:
        return id(self)

    def filter_tasks(
        self,
        task_names: list[str] | None = None,
        languages: list[str] | None = None,
        domains: list[TASK_DOMAIN] | None = None,
        task_types: list[TASK_TYPE] | None = None,  # type: ignore
        modalities: list[MODALITIES] | None = None,
    ) -> BenchmarkResults:
        # TODO: Same as filter_models
        model_results = [
            res.filter_tasks(
                task_names=task_names,
                languages=languages,
                domains=domains,
                task_types=task_types,
                modalities=modalities,
            )
            for res in self.model_results
        ]
        return type(self).model_construct(
            model_results=[res for res in model_results if res.task_results]
        )

    def select_tasks(self, tasks: Sequence[AbsTask]) -> BenchmarkResults:
        """Select tasks from the benchmark results.

        Args:
            tasks: List of tasks to select. Can be a list of AbsTask objects or task names.
        """
        new_model_results = [
            model_res.select_tasks(tasks) for model_res in self.model_results
        ]
        return type(self).model_construct(model_results=new_model_results)

    def select_models(
        self,
        names: list[str] | list[ModelMeta],
        revisions: list[str | None] | None = None,
    ) -> BenchmarkResults:
        """Get models by name and revision.

        Args:
            names: List of model names to filter by. Can also be a list of ModelMeta objects. In which case, the revision is ignored.
            revisions: List of model revisions to filter by. If None, all revisions are returned.
        """
        models_res = []
        _revisions = revisions if revisions is not None else [None] * len(names)

        name_rev = {}

        if len(names) != len(_revisions):
            raise ValueError(
                "The length of names and revisions must be the same or revisions must be None."
            )

        for name, revision in zip(names, _revisions):
            if isinstance(name, ModelMeta):
                name_rev[name.name] = name.revision
            else:
                name_rev[name] = revision

        for model_res in self.model_results:
            model_name = model_res.model_name
            revision = model_res.model_revision
            if model_name in name_rev:
                if name_rev[model_name] is None or revision == name_rev[model_name]:
                    models_res.append(model_res)

        return type(self).model_construct(model_results=models_res)

    def filter_models(
        self,
        model_names: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        open_weights: bool | None = None,
        frameworks: Iterable[str] | None = None,
        n_parameters_range: tuple[int | None, int | None] = (None, None),
        use_instructions: bool | None = None,
        zero_shot_on: list[AbsTask] | None = None,
    ) -> BenchmarkResults:
        # TODO: This seems like mostly a utility function for the benchmark
        # I would probably move the filtering of the models outside of this call. No need to call get_model_metas inside the filter.
        # interface would then be the same as the get_models function

        model_metas = get_model_metas(
            model_names=model_names,
            languages=languages,
            open_weights=open_weights,
            frameworks=frameworks,
            n_parameters_range=n_parameters_range,
            use_instructions=use_instructions,
            zero_shot_on=zero_shot_on,
        )
        models = {meta.name for meta in model_metas}
        # model_revision_pairs = {(meta.name, meta.revision) for meta in model_metas}
        new_model_results = []
        for model_res in self:
            if model_res.model_name in models:
                new_model_results.append(model_res)

        return type(self).model_construct(model_results=new_model_results)

    def join_revisions(self) -> BenchmarkResults:
        """Join revisions of the same model.

        In case of conflicts, the following rules are applied:
        - If the main revision is present, it is kept. The main revision is the defined in the models ModelMeta object.
        - If there is multiple revisions and some of them are None or na, they are filtered out.
        - If there is no main revision, we prefer the one run using the latest mteb version.
        """
        # TODO: In v2 we should probably have this be the default when loading. We could probably even reduce loading times by only loading what we need

        def parse_version(version_str: str) -> Version | None:
            try:
                return Version(version_str)
            except (InvalidVersion, TypeError):
                return None

        def keep_best(group: pd.DataFrame) -> pd.DataFrame:
            # Filtering out task_results where no scores are present
            group = group[group["has_scores"]]
            is_main_revision = group["revision"] == group["main_revision"]
            # If the main revision is present we select that
            if is_main_revision.sum() > 0:
                return group[is_main_revision].head(n=1)
            unique_revisions = group["revision"].unique()

            # ensure None/NA/"external" revisions is filtered out
            group.loc[group["revision"].isna(), "revision"] = "no_revision_available"
            group.loc[group["revision"] == "external", "revision"] = (
                "no_revision_available"
            )

            # Filtering out no_revision_available if other revisions are present
            if (len(unique_revisions) > 1) and (
                "no_revision_available" in unique_revisions
            ):
                group = group[group["revision"] != "no_revision_available"]
            # If there are any not-NA mteb versions, we select the latest one
            if group["mteb_version"].notna().any():
                group = group.dropna(subset=["mteb_version"])
                group = group.sort_values("mteb_version", ascending=False)
                return group.head(n=1)
            return group.head(n=1)

        records = []
        for model_result in self:
            for task_result in model_result.task_results:
                records.append(
                    dict(
                        model=model_result.model_name,
                        revision=model_result.model_revision,
                        task_name=task_result.task_name,
                        mteb_version=task_result.mteb_version,
                        task_result=task_result,
                        has_scores=bool(task_result.scores),
                    )
                )
        if not records:
            return BenchmarkResults.model_construct(model_results=[])
        task_df = pd.DataFrame.from_records(records)
        model_to_main_revision = {
            meta.name: meta.revision for meta in get_model_metas()
        }
        task_df["main_revision"] = task_df["model"].map(model_to_main_revision)  # type: ignore
        task_df["mteb_version"] = task_df["mteb_version"].map(parse_version)  # type: ignore
        task_df = (
            task_df.groupby(["model", "task_name"])
            .apply(keep_best)
            .reset_index(drop=True)
        )
        model_results = []
        for (model, model_revision), group in task_df.groupby(["model", "revision"]):
            model_result = ModelResult.model_construct(
                model_name=model,
                model_revision=model_revision,
                task_results=list(group["task_result"]),
            )
            model_results.append(model_result)
        return BenchmarkResults.model_construct(model_results=model_results)

    def get_scores(
        self,
        splits: list[Split] | None = None,
        languages: list[ISO_LANGUAGE | ISO_LANGUAGE_SCRIPT] | None = None,
        scripts: list[ISO_LANGUAGE_SCRIPT] | None = None,
        getter: Callable[[ScoresDict], Score] | None = None,
        aggregation: Callable[[list[Score]], Any] | None = None,
        format: Literal["wide", "long"] = "wide",
    ) -> list[dict]:
        # TODO: Convert to private function in v2
        entries = []
        if format == "wide":
            for model_res in self:
                try:
                    model_scores = model_res.get_scores(
                        splits=splits,
                        languages=languages,
                        scripts=scripts,
                        getter=getter,
                        aggregation=aggregation,
                        format="wide",
                    )
                    entries.append(
                        {
                            "model": model_res.model_name,
                            "revision": model_res.model_revision,
                            **model_scores,  # type: ignore
                        }
                    )
                except Exception as e:
                    warnings.warn(
                        f"Couldn't get scores for {model_res.model_name}({model_res.model_revision}), due to: {e}"
                    )
        if format == "long":
            for model_res in self:
                try:
                    entries.extend(
                        model_res.get_scores(
                            splits=splits,
                            languages=languages,
                            scripts=scripts,
                            getter=getter,
                            aggregation=aggregation,
                            format="long",
                        )
                    )
                except Exception as e:
                    warnings.warn(
                        f"Couldn't get scores for {model_res.model_name}({model_res.model_revision}), due to: {e}"
                    )
        return entries

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
                If there are multiple revisions for the same model, they will be joined using the `join_revisions` method.
            format: The format of the DataFrame. Can be one of:
                - "wide": The DataFrame will be of shape (number of tasks, number of models). Scores will be in the cells.
                - "long": The DataFrame will of length (number of tasks * number of model). Scores will be in columns.

        Returns:
            A DataFrame with the scores for all models and tasks.
        """
        bench_results = self
        if include_model_revision is False:
            bench_results = bench_results.join_revisions()

        scores_data = []
        for model_result in bench_results:
            scores_data.extend(model_result._get_score_for_table())

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

        # Aggregation
        return _aggregate_and_pivot(
            df,
            columns=_columns,
            aggregation_level=aggregation_level,
            aggregation_fn=aggregation_fn,
            format=format,
        )

    def __iter__(self):
        return iter(self.model_results)

    def __getitem__(self, index) -> ModelResult:
        return self.model_results[index]

    def to_legacy_dict(self) -> dict[str, dict[str, list[TaskResult]]]:
        # TODO: Make private or remove in v2
        res = defaultdict(dict)
        for model_res in self:
            res[model_res.model_name][model_res.model_revision] = model_res.task_results
        return res

    @classmethod
    def from_legacy_dict(cls, legacy: dict[str, dict[str, list[TaskResult]]]):
        # TODO: Make private or remove in v2
        model_results = []
        for model_name, revisions in legacy.items():
            for model_revision, results in revisions.items():
                model_results.append(
                    ModelResult(
                        model_name=model_name,
                        model_revision=model_revision,
                        task_results=results,
                    )
                )
        return cls(model_results=model_results)

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> BenchmarkResults:
        return cls.model_validate(data)

    def to_disk(self, path: Path | str) -> None:
        path = Path(path)
        with path.open("w") as out_file:
            out_file.write(self.model_dump_json(indent=2))

    @classmethod
    def from_validated(cls, **data) -> BenchmarkResults:
        model_results = []
        for model_res in data["model_results"]:
            model_results.append(ModelResult.from_validated(**model_res))
        return cls.model_construct(model_results=model_results)

    @classmethod
    def from_disk(cls, path: Path | str) -> BenchmarkResults:
        path = Path(path)
        with path.open() as in_file:
            data = json.loads(in_file.read())
        return cls.from_dict(data)

    @property
    def languages(self) -> list[str]:
        """Get all languages in the benchmark results.

        Returns:
            A list of languages in ISO 639-1 format.
        """
        langs = []
        for model_res in self.model_results:
            langs.extend(model_res.languages)
        return list(set(langs))

    @property
    def domains(self) -> list[str]:
        """Get all domains in the benchmark results.

        Returns:
            A list of domains in ISO 639-1 format.
        """
        ds = []
        for model_res in self.model_results:
            ds.extend(model_res.domains)
        return list(set(ds))

    @property
    def task_types(self) -> list[str]:
        """Get all task types in the benchmark results.

        Returns:
            A list of task types.
        """
        # TODO: V2: Task types vs task categories - we should probably be consistent
        ts = []
        for model_res in self.model_results:
            ts.extend(model_res.task_types)
        return list(set(ts))

    @property
    def task_names(self) -> list[str]:
        """Get all task names in the benchmark results.

        Returns:
            A list of task names.
        """
        names = []
        for model_res in self.model_results:
            names.extend(model_res.task_names)
        return list(set(names))

    @property
    def modalities(self) -> list[str]:
        """Get all modalities in the benchmark results.

        Returns:
            A list of modalities.
        """
        mod = []
        for model_res in self.model_results:
            mod.extend(model_res.modalities)
        return list(set(mod))

    @property
    def model_names(self) -> list[str]:
        """Get all model names in the benchmark results.

        Returns:
            A list of model names.
        """
        return [model_res.model_name for model_res in self.model_results]

    @property
    def model_revisions(self) -> list[dict[str, str | None]]:
        """Get all model revisions in the benchmark results.

        Returns:
            A list of dictionaries with model names and revisions.
        """
        return [
            {"model_name": model_res.model_name, "revision": model_res.model_revision}
            for model_res in self.model_results
        ]
