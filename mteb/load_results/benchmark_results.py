from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict

from mteb.abstasks.AbsTask import AbsTask, ScoresDict
from mteb.abstasks.TaskMetadata import (
    ISO_LANGUAGE_SCRIPT,
    TASK_DOMAIN,
    TASK_TYPE,
)
from mteb.languages import ISO_LANGUAGE
from mteb.load_results.task_results import TaskResult
from mteb.models.overview import get_model_metas

Split = str
Score = Any


class ModelResult(BaseModel):
    model_name: str
    model_revision: str | None
    task_results: list[TaskResult]
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    def __repr__(self) -> str:
        n_entries = len(self.task_results)
        return f"ModelResult(model_name={self.model_name}, model_revision={self.model_revision}, task_results=[...](#{n_entries}))"

    def filter_tasks(
        self,
        task_names: list[str] | None = None,
        languages: list[str] | None = None,
        domains: list[TASK_DOMAIN] | None = None,
        task_types: list[TASK_TYPE] | None = None,
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
            new_task_results.append(task_result)
        return type(self)(
            model_name=self.model_name,
            model_revision=self.model_revision,
            task_results=new_task_results,
        )

    def select_tasks(self, tasks: list[AbsTask]) -> ModelResult:
        task_name_to_task = {task.metadata.name: task for task in tasks}
        new_task_results = [
            task_res.validate_and_filter_scores(task_name_to_task[task_res.task_name])
            for task_res in self.task_results
            if task_res.task_name in task_name_to_task
        ]
        return type(self)(
            model_name=self.model_name,
            model_revision=self.model_revision,
            task_results=new_task_results,
        )

    def get_scores(
        self,
        splits: list[Split] | None = None,
        languages: list[ISO_LANGUAGE | ISO_LANGUAGE_SCRIPT] | None = None,
        scripts: list[ISO_LANGUAGE_SCRIPT] | None = None,
        getter: Callable[[ScoresDict], Score] = lambda scores: scores["main_score"],
        aggregation: Callable[[list[Score]], Any] = np.mean,
        format: Literal["wide", "long"] = "wide",
    ) -> dict | list:
        if format == "wide":
            scores = {
                res.task_name: res.get_score(
                    splits=splits,
                    languages=languages,
                    scripts=scripts,
                    getter=getter,
                    aggregation=aggregation,
                )
                for res in self.task_results
            }
            return scores
        if format == "long":
            entries = []
            for task_res in self.task_results:
                entry = dict(  # noqa
                    model_name=self.model_name,
                    model_revision=self.model_revision,
                    task_name=task_res.task_name,
                    score=task_res.get_score(
                        splits=splits,
                        languages=languages,
                        getter=getter,
                        aggregation=aggregation,
                    ),
                    mteb_version=task_res.mteb_version,
                    dataset_revision=task_res.dataset_revision,
                    evaluation_time=task_res.evaluation_time,
                    kg_co2_emissions=task_res.kg_co2_emissions,
                )
                entries.append(entry)
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


class BenchmarkResults(BaseModel):
    model_results: list[ModelResult]
    model_config = ConfigDict(
        protected_namespaces=(),
    )

    def __repr__(self) -> str:
        n_models = len(self.model_results)
        return f"BenchmarkResults(model_results=[...](#{n_models}))"

    def filter_tasks(
        self,
        task_names: list[str] | None = None,
        languages: list[str] | None = None,
        domains: list[TASK_DOMAIN] | None = None,
        task_types: list[TASK_TYPE] | None = None,
    ) -> BenchmarkResults:
        model_results = [
            res.filter_tasks(
                task_names=task_names,
                languages=languages,
                domains=domains,
                task_types=task_types,
            )
            for res in self.model_results
        ]
        return type(self)(
            model_results=[res for res in model_results if res.task_results]
        )

    def select_tasks(self, tasks: list[AbsTask]) -> BenchmarkResults:
        new_model_results = [
            model_res.select_tasks(tasks) for model_res in self.model_results
        ]
        return type(self)(model_results=new_model_results)

    def filter_models(
        self,
        model_names: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        open_source: bool | None = None,
        frameworks: Iterable[str] | None = None,
        n_parameters_range: tuple[int | None, int | None] = (None, None),
    ) -> BenchmarkResults:
        model_metas = get_model_metas(
            model_names, languages, open_source, frameworks, n_parameters_range
        )
        model_revision_pairs = {(meta.name, meta.revision) for meta in model_metas}
        new_model_results = []
        for model_res in self:
            if (model_res.model_name, model_res.model_revision) in model_revision_pairs:
                new_model_results.append(model_res)
        return type(self)(model_results=new_model_results)

    def get_scores(
        self,
        splits: list[Split] | None = None,
        languages: list[ISO_LANGUAGE | ISO_LANGUAGE_SCRIPT] | None = None,
        scripts: list[ISO_LANGUAGE_SCRIPT] | None = None,
        getter: Callable[[ScoresDict], Score] = lambda scores: scores["main_score"],
        aggregation: Callable[[list[Score]], Any] = np.mean,
        format: Literal["wide", "long"] = "wide",
    ) -> list[dict]:
        entries = []
        if format == "wide":
            for model_res in self:
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
        if format == "long":
            for model_res in self:
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
    def from_dict(cls, data: dict) -> TaskResult:
        return cls.model_validate(data)

    def to_disk(self, path: Path | str) -> None:
        path = Path(path)
        with path.open("w") as out_file:
            out_file.write(self.model_dump_json(indent=2))

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
