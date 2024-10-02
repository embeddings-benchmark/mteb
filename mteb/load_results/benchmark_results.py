import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from mteb.abstasks.AbsTask import AbsTask, ScoresDict
from mteb.abstasks.TaskMetadata import (ISO_LANGUAGE_SCRIPT, TASK_CATEGORY,
                                        TASK_DOMAIN, TASK_TYPE)
from mteb.languages import ISO_LANGUAGE
from mteb.load_results.task_results import TaskResult
from mteb.models.overview import get_model_metas
from mteb.overview import get_tasks


def restrict_task_results(res: TaskResult, task: AbsTask) -> TaskResult:
    splits = task.metadata.eval_splits
    hf_subsets = set(task.metadata.hf_subsets_to_langscripts)
    new_scores = {}
    seen_splits = set()
    for split in res.scores:
        if split not in splits:
            continue
        new_scores[split] = []
        seen_subsets = set()
        for _scores in res.scores[split]:
            if _scores["hf_subset"] not in hf_subsets:
                continue
            new_scores[split].append(_scores)
            seen_subsets.add(_scores["hf_subset"])
        if seen_subsets != hf_subsets:
            raise ValueError(
                f"Missing subsets {hf_subsets - seen_subsets} for split {split}"
            )
        seen_splits.add(split)
    if seen_splits != set(splits):
        raise ValueError(f"Missing splits {set(splits) - seen_splits}")
    new_res = {**res.to_dict(), "scores": new_scores}
    new_res = TaskResult.from_dict(new_res)
    return new_res


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
        languages: list[str] | None = None,
        script: list[str] | None = None,
        domains: list[TASK_DOMAIN] | None = None,
        task_types: list[TASK_TYPE] | None = None,
        categories: list[TASK_CATEGORY] | None = None,
        tasks: list[str] | None = None,
        exclude_superseeded: bool = True,
    ) -> "ModelResult":
        filtered_tasks = get_tasks(
            languages=languages,
            script=script,
            domains=domains,
            task_types=task_types,
            categories=categories,
            tasks=tasks,
            exclude_superseeded=exclude_superseeded,
        )
        return self.select_tasks(filtered_tasks)

    def select_tasks(self, tasks: list[AbsTask]) -> "ModelResult":
        tasks = {task.metadata.name: task for task in tasks}
        new_task_results = [
            restrict_task_results(res, tasks[res.task_name])
            for res in self.task_results
            if res.task_name in tasks
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
    ) -> dict[str, float]:
        return {
            res.task_name: res.get_score(
                splits=splits,
                languages=languages,
                scripts=scripts,
                getter=getter,
                aggregation=aggregation,
            )
            for res in self.task_results
        }

    def __iter__(self):
        return iter(self.task_results)

    def __getitem__(self, index) -> TaskResult:
        return self.task_results[index]


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
        languages: list[str] | None = None,
        script: list[str] | None = None,
        domains: list[TASK_DOMAIN] | None = None,
        task_types: list[TASK_TYPE] | None = None,
        categories: list[TASK_CATEGORY] | None = None,
        tasks: list[str] | None = None,
        exclude_superseeded: bool = True,
    ) -> "BenchmarkResults":
        model_results = [
            res.filter_tasks(
                languages=languages,
                script=script,
                domains=domains,
                task_types=task_types,
                categories=categories,
                tasks=tasks,
                exclude_superseeded=exclude_superseeded,
            )
            for res in self.model_results
        ]
        return type(self)(
            model_results=[res for res in model_results if res.task_results]
        )

    def select_tasks(self, tasks: list[AbsTask]) -> "BenchmarkResults":
        model_results = [res.select_tasks(tasks) for res in self.model_results]
        return type(self)(
            model_results=[res for res in model_results if res.task_results]
        )

    def filter_models(
        self,
        model_names: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        open_source: bool | None = None,
        frameworks: Iterable[str] | None = None,
        n_parameters_range: tuple[int | None, int | None] = (None, None),
    ) -> "BenchmarkResults":
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
    ) -> list[dict[str, Any]]:
        res = []
        for model_res in self:
            model_scores = model_res.get_scores(
                splits=splits,
                languages=languages,
                scripts=scripts,
                getter=getter,
                aggregation=aggregation,
            )
            res.append(
                {
                    "model": model_res.model_name,
                    "revision": model_res.model_revision,
                    **model_scores,
                }
            )
        return res

    def to_table(
        self,
        splits: list[Split] | None = None,
        languages: list[ISO_LANGUAGE | ISO_LANGUAGE_SCRIPT] | None = None,
        scripts: list[ISO_LANGUAGE_SCRIPT] | None = None,
        getter: Callable[[ScoresDict], Score] = lambda scores: scores["main_score"],
        aggregation: Callable[[list[Score]], Any] = np.mean,
        format: Literal["wide", "long"] = "wide",
    ) -> pd.DataFrame:
        if format == "wide":
            entries = self.get_scores(
                splits=splits,
                languages=languages,
                getter=getter,
                aggregation=aggregation,
            )
            return pd.DataFrame(entries).set_index(["model_name", "model_revision"])
        elif format == "long":
            entries = []
            for model_res in self:
                for task_res in model_res:
                    entry = dict(
                        model_name=model_res.model_name,
                        model_revision=model_res.model_revision,
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
            return pd.DataFrame(entries)
        else:
            raise ValueError(
                f"Table format can either be 'long' or 'wide', not {format}"
            )

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
    def from_disk(cls, path: Path | str) -> "BenchmarkResults":
        path = Path(path)
        with path.open() as in_file:
            data = json.loads(in_file.read())
        return cls.from_dict(data)
