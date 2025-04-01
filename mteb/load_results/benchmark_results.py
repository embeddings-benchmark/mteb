from __future__ import annotations

import json
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
from mteb.models.overview import get_model_metas

Split = str
Score = Any


class ModelResult(BaseModel):
    model_name: str
    model_revision: str | None
    task_results: list[TaskResult]
    default_modalities: list[MODALITIES] = Field(
        default_factory=lambda: ["text"], alias="modalities"
    )
    model_config = ConfigDict(
        protected_namespaces=(),
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
                            splits=splits, languages=languages
                        )
                    else:
                        score = task_res.get_score(
                            splits=splits,
                            languages=languages,
                            aggregation=aggregation,
                            getter=getter,
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

    def __iter__(self):
        return iter(self.task_results)

    def __getitem__(self, index) -> TaskResult:
        return self.task_results[index]

    @property
    def languages(self) -> list[str]:
        langs = []
        for task_res in self.task_results:
            langs.extend(task_res.languages)
        return list(set(langs))

    @property
    def domains(self) -> list[str]:
        ds = []
        for task_res in self.task_results:
            ds.extend(task_res.domains)
        return list(set(ds))

    @property
    def task_types(self) -> list[str]:
        return list({task_res.task_type for task_res in self.task_results})

    @property
    def task_names(self) -> list[str]:
        return [task_res.task_name for task_res in self.task_results]

    @property
    def modalities(self) -> list[str]:
        mods = []
        for task_res in self.task_results:
            task_modalities = getattr(task_res, "modalities", [])
            mods.extend(task_modalities)
        if not mods:
            mods = self.default_modalities
        return list(set(mods))


class BenchmarkResults(BaseModel):
    model_results: list[ModelResult]
    model_config = ConfigDict(
        protected_namespaces=(),
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
        task_types: list[TASK_TYPE] | None = None,
        modalities: list[MODALITIES] | None = None,
    ) -> BenchmarkResults:
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
        new_model_results = [
            model_res.select_tasks(tasks) for model_res in self.model_results
        ]
        return type(self).model_construct(model_results=new_model_results)

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
        # if model_names is None:
        #     model_names = [model_res.model_name for model_res in self]
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

    def join_revisions(self):
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
            for task_result in model_result:
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
        task_df["main_revision"] = task_df["model"].map(model_to_main_revision)
        task_df["mteb_version"] = task_df["mteb_version"].map(parse_version)
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
                            **model_scores,
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

    def __iter__(self):
        return iter(self.model_results)

    def __getitem__(self, index) -> ModelResult:
        return self.model_results[index]

    def to_legacy_dict(self) -> dict[str, dict[str, list[TaskResult]]]:
        res = defaultdict(dict)
        for model_res in self:
            res[model_res.model_name][model_res.model_revision] = model_res.task_results
        return res

    @classmethod
    def from_legacy_dict(cls, legacy: dict[str, dict[str, list[TaskResult]]]):
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
        langs = []
        for model_res in self.model_results:
            langs.extend(model_res.languages)
        return list(set(langs))

    @property
    def domains(self) -> list[str]:
        ds = []
        for model_res in self.model_results:
            ds.extend(model_res.domains)
        return list(set(ds))

    @property
    def task_types(self) -> list[str]:
        ts = []
        for model_res in self.model_results:
            ts.extend(model_res.task_types)
        return list(set(ts))

    @property
    def task_names(self) -> list[str]:
        names = []
        for model_res in self.model_results:
            names.extend(model_res.task_names)
        return list(set(names))

    @property
    def modalities(self) -> list[str]:
        mod = []
        for model_res in self.model_results:
            mod.extend(model_res.modalities)
        return list(set(mod))
