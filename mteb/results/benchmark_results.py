import json
import logging
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, ConfigDict
from typing_extensions import Self

from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import (
    TaskDomain,
    TaskType,
)
from mteb.models import ModelMeta
from mteb.models.get_model_meta import get_model_metas
from mteb.types import (
    ISOLanguage,
    ISOLanguageScript,
    Modalities,
    Score,
    ScoresDict,
    SplitName,
)

from .model_result import ModelResult, _aggregate_and_pivot

logger = logging.getLogger(__name__)


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

    def _filter_tasks(
        self,
        task_names: list[str] | None = None,
        languages: list[str] | None = None,
        domains: list[TaskDomain] | None = None,
        task_types: list[TaskType] | None = None,  # type: ignore
        modalities: list[Modalities] | None = None,
        is_public: bool | None = None,
    ) -> Self:
        # TODO: Same as filter_models
        model_results = [
            res._filter_tasks(
                task_names=task_names,
                languages=languages,
                domains=domains,
                task_types=task_types,
                modalities=modalities,
                is_public=is_public,
            )
            for res in self.model_results
        ]
        return type(self).model_construct(
            model_results=[res for res in model_results if res.task_results]
        )

    def select_tasks(self, tasks: Sequence[AbsTask]) -> Self:
        """Select tasks from the benchmark results.

        Args:
            tasks: List of tasks to select. Can be a list of AbsTask objects or task names.

        Returns:
            A new BenchmarkResults object with the selected tasks.
        """
        new_model_results = [
            model_res.select_tasks(tasks) for model_res in self.model_results
        ]
        return type(self).model_construct(model_results=new_model_results)

    def select_models(
        self,
        names: list[str] | list[ModelMeta],
        revisions: list[str | None] | None = None,
    ) -> Self:
        """Get models by name and revision.

        Args:
            names: List of model names to filter by. Can also be a list of ModelMeta objects. In which case, the revision is ignored.
            revisions: List of model revisions to filter by. If None, all revisions are returned.

        Returns:
            A new BenchmarkResults object with the filtered models.
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

    def _filter_models(
        self,
        model_names: Iterable[str] | None = None,
        languages: Iterable[str] | None = None,
        open_weights: bool | None = None,
        frameworks: Iterable[str] | None = None,
        n_parameters_range: tuple[int | None, int | None] = (None, None),
        use_instructions: bool | None = None,
        zero_shot_on: list[AbsTask] | None = None,
    ) -> Self:
        # mostly a utility function for the leaderboard app.
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

    def join_revisions(self) -> Self:
        """Join revisions of the same model.

        In case of conflicts, the following rules are applied:
        1) If the main revision is present, it is kept. The main revision is the defined in the models ModelMeta object.
        2) If there is multiple revisions and some of them are None or na, they are filtered out.
        3) If there is no main revision, we prefer the one run using the latest mteb version.

        Returns:
            A new BenchmarkResults object with the revisions joined.
        """

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

    def _get_scores(
        self,
        splits: list[SplitName] | None = None,
        languages: list[ISOLanguage | ISOLanguageScript] | None = None,
        scripts: list[ISOLanguageScript] | None = None,
        getter: Callable[[ScoresDict], Score] | None = None,
        aggregation: Callable[[list[Score]], Any] | None = None,
        format: Literal["wide", "long"] = "wide",
    ) -> list[dict]:
        entries = []
        if format == "wide":
            for model_res in self:
                try:
                    model_scores = model_res._get_scores(
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
                        model_res._get_scores(
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
        aggregation_level: Literal["subset", "split", "task", "language"] = "task",
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

        Afterward, the DataFrame will be aggregated according to the aggregation method and pivoted to either a wide format.

        Args:
            aggregation_level: The aggregation to use. Can be one of:
                - "subset"/None: No aggregation will be done. The DataFrame will have one row per model, task, split and subset.
                - "split": Aggregates the scores by split. The DataFrame will have one row per model, task and split.
                - "task": Aggregates the scores by task. The DataFrame will have one row per model and task.
                - "language": Aggregates the scores by language. The DataFrame will have one row per model and language.
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

    def __iter__(self) -> Iterator[ModelResult]:
        return iter(self.model_results)

    def __getitem__(self, index: int) -> ModelResult:
        return self.model_results[index]

    def to_dict(self) -> dict:
        """Convert BenchmarkResults to a dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create BenchmarkResults from a dictionary."""
        return cls.model_validate(data)

    def to_disk(self, path: Path | str) -> None:
        """Save the BenchmarkResults to a JSON file."""
        path = Path(path)
        with path.open("w") as out_file:
            out_file.write(self.model_dump_json(indent=2))

    @classmethod
    def from_validated(cls, **data) -> Self:
        """Create BenchmarkResults from validated data.

        Args:
            data: Dictionary containing the data.

        Returns:
            An instance of BenchmarkResults.
        """
        model_results = []
        for model_res in data["model_results"]:
            model_results.append(ModelResult.from_validated(**model_res))
        return cls.model_construct(model_results=model_results)

    @classmethod
    def from_disk(cls, path: Path | str) -> Self:
        """Load the BenchmarkResults from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            An instance of BenchmarkResults.
        """
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
