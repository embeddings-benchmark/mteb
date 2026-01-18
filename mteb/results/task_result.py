from __future__ import annotations

import json
import logging
import warnings
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import EvalResult
from packaging.version import Version
from pydantic import BaseModel, field_validator
from typing_extensions import Self

from mteb import TaskMetadata
from mteb._helpful_enum import HelpfulStrEnum
from mteb.abstasks import AbsTaskClassification
from mteb.abstasks.abstask import AbsTask
from mteb.abstasks.task_metadata import TaskDomain
from mteb.languages import LanguageScripts
from mteb.models.model_meta import ScoringFunction
from mteb.types import (
    HFSubset,
    ISOLanguage,
    ISOLanguageScript,
    Score,
    ScoresDict,
    SplitName,
)

logger = logging.getLogger(__name__)


class Criteria(HelpfulStrEnum):
    """Enum for criteria to check when merging TaskResult objects."""

    MTEB_VERSION = "mteb_version"
    DATASET_REVISION = "dataset_revision"


class ScalaNbClassificationDummy(AbsTaskClassification):
    """A dummy task for loading historic results from before v1.11.0"""

    metadata = TaskMetadata(
        name="ScalaNbClassification",
        description="A dummy",
        main_score="accuracy",
        type="Classification",
        eval_langs=["nob-Latn"],
        dataset={"path": "not/exists", "revision": "revision_not_applicable"},
    )


class ScalaNnClassificationDummy(AbsTaskClassification):
    """A dummy task for loading historic results from before v1.11.0"""

    metadata = TaskMetadata(
        name="ScalaNnClassification",
        description="A dummy",
        main_score="accuracy",
        type="Classification",
        eval_langs=["nob-Latn"],
        dataset={"path": "not/exists", "revision": "revision_not_applicable"},
    )


class ScalaDaClassificationDummy(AbsTaskClassification):
    """A dummy task for loading historic results from before v1.11.0"""

    metadata = TaskMetadata(
        name="ScalaDaClassification",
        description="A dummy",
        main_score="accuracy",
        type="Classification",
        eval_langs=["dan-Latn"],
        dataset={"path": "not/exists", "revision": "revision_not_applicable"},
    )


class ScalaSvClassificationDummy(AbsTaskClassification):
    """A dummy task for loading historic results from before v1.11.0"""

    metadata = TaskMetadata(
        name="ScalaSvClassification",
        description="A dummy",
        main_score="accuracy",
        type="Classification",
        eval_langs=["swe-Latn"],
        dataset={"path": "not/exists", "revision": "revision_not_applicable"},
    )


outdated_tasks: dict[str, type[AbsTask]] = {
    "ScalaNbClassification": ScalaNbClassificationDummy,
    "ScalaNnClassification": ScalaNnClassificationDummy,
    "ScalaDaClassification": ScalaDaClassificationDummy,
    "ScalaSvClassification": ScalaSvClassificationDummy,
}

renamed_tasks = {
    "NorwegianParliament": "NorwegianParliamentClassification",
    "CMedQAv2": "CMedQAv2-reranking",
    "CMedQAv1": "CMedQAv1-reranking",
    "8TagsClustering": "EightTagsClustering",
    "PPC": "PpcPC",
}


class TaskResult(BaseModel):
    """A class to represent the MTEB result.

    Attributes:
        task_name: The name of the MTEB task.
        dataset_revision: The revision dataset for the task on HuggingFace dataset hub.
        mteb_version: The version of the MTEB used to evaluate the model.
        scores: The scores of the model on the dataset. The scores is a dictionary with the following structure; dict[SplitName, list[Scores]].
            Where Scores is a dictionary with the following structure; dict[str, Any]. Where the keys and values are scores. Split is the split of
            the dataset.
        evaluation_time: The time taken to evaluate the model.
        kg_co2_emissions: The kg of CO2 emissions produced by the model during evaluation.

    Examples:
        >>> scores = {
        ...     "evaluation_time": 100,
        ...     "train": {
        ...         "en-de": {
        ...             "main_score": 0.5,
        ...         },
        ...         "en-fr": {
        ...             "main_score": 0.6,
        ...         },
        ...     },
        ... }
        >>> sample_task = ... # some MTEB task
        >>> mteb_results = TaskResult.from_task_results(sample_task, scores)
        >>> mteb_results.get_score()  # get the main score for all languages
        0.55
        >>> mteb_results.get_score(languages=["fra"])  # get the main score for French
        0.6
        >>> mteb_results.to_dict()
        {'dataset_revision': '1.0', 'task_name': 'sample_task', 'mteb_version': '1.0.0', 'evaluation_time': 100, 'scores': {'train':
            [
                {'main_score': 0.5, 'hf_subset': 'en-de', 'languages': ['eng-Latn', 'deu-Latn']},
                {'main_score': 0.6, 'hf_subset': 'en-fr', 'languages': ['eng-Latn', 'fra-Latn']}
            ]}
        }
    """

    dataset_revision: str
    task_name: str
    mteb_version: str | None
    scores: dict[SplitName, list[ScoresDict]]
    evaluation_time: float | None
    kg_co2_emissions: float | None = None

    @classmethod
    def from_task_results(
        cls,
        task: AbsTask | type[AbsTask],
        scores: dict[SplitName, Mapping[HFSubset, ScoresDict]],
        evaluation_time: float,
        kg_co2_emissions: float | None = None,
    ) -> TaskResult:
        """Create a TaskResult from the task and scores.

        Args:
            task: The task to create the TaskResult from.
            scores: The scores of the model on the dataset. The scores is a dictionary with the following structure; dict[SplitName, dict[HFSubset, Scores]].
                Where Scores is a dictionary with the following structure; dict[str, Any]. Where the keys and values are scores. Split is the split of
                the dataset.
            evaluation_time: The time taken to evaluate the model.
            kg_co2_emissions: The kg of CO2 emissions produced by the model during evaluation.
        """
        task_meta = task.metadata
        subset2langscripts = task_meta.hf_subsets_to_langscripts
        flat_scores = defaultdict(list)
        for split, hf_subset_scores in scores.items():
            for hf_subset, hf_scores in hf_subset_scores.items():
                eval_langs = subset2langscripts[hf_subset]
                _scores = {
                    **hf_scores,
                    "hf_subset": hf_subset,
                    "languages": eval_langs,
                }
                flat_scores[split].append(_scores)

        return TaskResult(
            dataset_revision=task.metadata.revision,
            task_name=task.metadata.name,
            mteb_version=version("mteb"),
            scores=flat_scores,
            evaluation_time=evaluation_time,
            kg_co2_emissions=kg_co2_emissions,
        )

    @field_validator("scores")
    @classmethod
    def _validate_scores(
        cls, v: dict[SplitName, list[ScoresDict]]
    ) -> dict[SplitName, list[ScoresDict]]:
        for split, hf_subset_scores in v.items():
            for hf_subset_score in hf_subset_scores:
                if not isinstance(hf_subset_score, dict):
                    raise ValueError("Scores should be a dictionary")
                cls._validate_scores_dict(hf_subset_score)
        return v

    @staticmethod
    def _validate_scores_dict(scores: ScoresDict) -> None:
        if "main_score" not in scores:
            raise ValueError("'main_score' should be in scores")
        if "hf_subset" not in scores or not isinstance(scores["hf_subset"], str):
            raise ValueError("hf_subset should be in scores and should be a string")
        if "languages" not in scores or not isinstance(scores["languages"], list):
            raise ValueError("languages should be in scores and should be a list")

        # check that it is json serializable
        try:
            _ = json.dumps(scores)
        except Exception as e:
            raise ValueError(f"Scores are not json serializable: {e}")

    @property
    def languages(self) -> list[str]:
        """Get the languages present in the scores."""
        langs = []
        for split, split_res in self.scores.items():
            for entry in split_res:
                langs.extend([lang.split("-")[0] for lang in entry["languages"]])
        return list(set(langs))

    @cached_property
    def task(self) -> AbsTask:
        """Get the task associated with the result."""
        from mteb.get_tasks import get_task

        return get_task(self.task_name)

    @property
    def domains(self) -> list[TaskDomain]:
        """Get the domains of the task."""
        doms = self.task.metadata.domains
        if doms is None:
            doms = []
        return doms

    @property
    def task_type(self) -> str:
        """Get the type of the task."""
        return self.task.metadata.type

    @property
    def is_public(self) -> bool:
        """Check if the task is public."""
        return self.task.metadata.is_public

    @property
    def hf_subsets(self) -> list[str]:
        """Get the hf_subsets present in the scores."""
        hf_subsets = set()
        for split, split_res in self.scores.items():
            for entry in split_res:
                hf_subsets.add(entry["hf_subset"])
        return list(hf_subsets)

    @property
    def eval_splits(self) -> list[str]:
        """Get the eval splits present in the scores."""
        return list(self.scores.keys())

    def to_dict(self) -> dict:
        """Convert the TaskResult to a dictionary.

        Returns:
            The TaskResult as a dictionary.
        """
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Create a TaskResult from a dictionary.

        Args:
            data: The dictionary to create the TaskResult from.

        Returns:
            The created TaskResult object.
        """
        return cls.model_validate(data)

    def _round_scores(self, scores: dict[SplitName, list[ScoresDict]], n: int) -> None:
        """Recursively round scores to n decimal places"""
        for key, value in scores.items():
            if isinstance(value, dict):
                self._round_scores(value, n)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, dict):
                        self._round_scores(v, n)
                    elif isinstance(v, float):
                        value[i] = round(v, n)  # type: ignore[call-overload]

            elif isinstance(value, float):
                scores[key] = round(value, n)

    def to_disk(self, path: Path) -> None:
        """Save TaskResult to disk.

        Args:
            path: The path to the file to save.
        """
        json_obj = self.model_dump()
        self._round_scores(json_obj["scores"], 6)

        with path.open("w") as f:
            json.dump(json_obj, f, indent=2)

    @classmethod
    def from_disk(cls, path: Path, load_historic_data: bool = True) -> TaskResult:
        """Load TaskResult from disk.

        Args:
            path: The path to the file to load.
            load_historic_data: Whether to attempt to load historic data from before v1.11.0.

        Returns:
            The loaded TaskResult object.
        """
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not load_historic_data:
            try:
                return cls.model_validate(data)
            except Exception as e:
                raise ValueError(
                    f"Error loading TaskResult from disk. You can try to load historic data by setting `load_historic_data=True`. Error: {e}"
                )

        pre_1_11_load = (
            (
                "mteb_version" in data
                and data["mteb_version"] is not None
                and Version(data["mteb_version"]) < Version("1.11.0")
            )
            or "mteb_version" not in data
        )  # assume it is before 1.11.0 if the version is not present

        try:
            obj: TaskResult = cls.model_validate(data)
        except Exception as e:
            if not pre_1_11_load:
                raise e
            logger.debug(
                f"Could not load TaskResult from disk, got error: {e}. Attempting to load from disk using format from before v1.11.0"
            )
            obj = cls._convert_from_before_v1_11_0(data)

        pre_v_12_48 = (
            "mteb_version" in data
            and data["mteb_version"] is not None
            and Version(data["mteb_version"]) < Version("1.12.48")
        )

        if pre_v_12_48:
            cls._fix_pair_classification_scores(obj)

        return obj

    @classmethod
    def _fix_pair_classification_scores(cls, obj: TaskResult) -> None:
        from mteb import get_task

        task_name = obj.task_name
        task: AbsTask | type[AbsTask]
        if task_name in outdated_tasks:
            task = outdated_tasks[task_name]
        else:
            task = get_task(obj.task_name)

        if task.metadata.type == "PairClassification":
            for split, split_scores in obj.scores.items():
                for hf_subset_scores in split_scores:
                    # concatenate score e.g. ["max"]["ap"] -> ["max_ap"]
                    for key in list(hf_subset_scores.keys()):
                        if isinstance(hf_subset_scores[key], dict):
                            for k, v in hf_subset_scores[key].items():
                                hf_subset_scores[f"{key}_{k}"] = v  # type: ignore[index]
                            hf_subset_scores.pop(key)  # type: ignore[attr-defined]

    @classmethod
    def _convert_from_before_v1_11_0(cls, data: dict) -> TaskResult:
        from mteb.get_tasks import _TASKS_REGISTRY

        # in case the task name is not found in the registry, try to find a lower case version
        lower_case_registry = {k.lower(): v for k, v in _TASKS_REGISTRY.items()}

        scores = {**data}

        dataset_revision = scores.pop(
            "dataset_revision", "dataset revision not available"
        )
        task_name = scores.pop("mteb_dataset_name")
        mteb_version = scores.pop("mteb_version", "mteb version not available")

        # calculate evaluation time across all splits (move to top level)
        evaluation_time = 0
        for split, split_score in scores.items():
            if "evaluation_time" in split_score:
                evaluation_time += split_score.pop("evaluation_time")

        # normalize the scores to always be {split: {hf_subset: scores}}
        contains_hf_subset = any(
            isinstance(hf_subset_scores, dict)
            for split_scores in scores.values()
            for k, hf_subset_scores in split_scores.items()
            if k
            not in {"v_measures", "cos_sim", "euclidean", "manhattan", "dot", "max"}
        )
        if not contains_hf_subset:
            for split, split_score in scores.items():
                scores[split] = {"default": split_score.copy()}

        if task_name in outdated_tasks:
            logger.debug(
                f"Loading {task_name} as a dummy task as it no longer exists within MTEB. To avoid this set `load_historic_data=False`"
            )
            task = outdated_tasks[task_name]
        else:
            if task_name in renamed_tasks:
                task_name = renamed_tasks[task_name]
            task = _TASKS_REGISTRY.get(
                task_name, lower_case_registry[task_name.lower()]
            )

        # make sure that main score exists
        main_score = task.metadata.main_score
        for split, split_score in scores.items():
            for hf_subset, hf_subset_scores in split_score.items():
                for name, prev_name in [
                    (ScoringFunction.COSINE.value, "cos_sim"),
                    (ScoringFunction.MANHATTAN.value, "manhattan"),
                    (ScoringFunction.EUCLIDEAN.value, "euclidean"),
                    (ScoringFunction.DOT_PRODUCT.value, "dot"),
                    ("max", "max"),
                    ("similarity", "similarity"),
                ]:
                    prev_name_scores = hf_subset_scores.pop(prev_name, None)
                    if prev_name_scores is not None:
                        for k, v in prev_name_scores.items():
                            hf_subset_scores[f"{name}_{k}"] = v

                if "main_score" not in hf_subset_scores:
                    if main_score in hf_subset_scores:
                        hf_subset_scores["main_score"] = hf_subset_scores[main_score]
                    else:
                        msg = f"Main score {main_score} not found in scores"
                        logger.warning(msg)
                        warnings.warn(msg)
                        hf_subset_scores["main_score"] = None

        # specific fixes:
        if task_name == "MLSUMClusteringP2P" and mteb_version in [
            "1.1.2.dev0",
            "1.1.3.dev0",
        ]:  # back then it was only the french subsection which was implemented
            scores["test"]["fr"] = scores["test"].pop("default")
        if task_name == "MLSUMClusteringS2S" and mteb_version in [
            "1.1.2.dev0",
            "1.1.3.dev0",
        ]:
            scores["test"]["fr"] = scores["test"].pop("default")
        if task_name == "XPQARetrieval":  # subset were renamed from "fr" to "fra-fra"
            if "test" in scores and "fr" in scores["test"]:
                scores["test"]["fra-fra"] = scores["test"].pop("fr")

        result: TaskResult = TaskResult.from_task_results(
            task,
            scores,
            evaluation_time,
            kg_co2_emissions=None,
        )
        result.dataset_revision = dataset_revision
        result.mteb_version = mteb_version
        return result

    def get_score(
        self,
        splits: list[SplitName] | None = None,
        languages: list[ISOLanguage | ISOLanguageScript] | None = None,
        scripts: list[ISOLanguageScript] | None = None,
        getter: Callable[[ScoresDict], Score] = lambda scores: scores["main_score"],
        aggregation: Callable[[list[Score]], Any] = np.mean,
    ) -> Any:
        """Get a score for the specified splits, languages, scripts and aggregation function.

        Args:
            splits: The splits to consider.
            languages: The languages to consider. Can be ISO language codes or ISO language script codes.
            scripts: The scripts to consider.
            getter: A function that takes a scores dictionary and returns a score e.g. "main_score" or "evaluation_time".
            aggregation: The aggregation function to use.

        Returns:
            The result of the aggregation function on the scores.
        """
        if splits is None:
            splits = list(self.scores.keys())

        lang_scripts = LanguageScripts.from_languages_and_scripts(languages, scripts)

        values = []
        for split in splits:
            if split not in self.scores:
                raise ValueError(f"Split {split} not found in scores")

            for scores in self.scores[split]:
                eval_langs = scores["languages"]
                for lang in eval_langs:
                    if lang_scripts.contains_language(lang):
                        values.append(getter(scores))
                        break

        return aggregation(values)

    def _get_score_fast(
        self,
        splits: Iterable[str] | None = None,
        languages: list[ISOLanguage | ISOLanguageScript] | None = None,
        subsets: Iterable[str] | None = None,
    ) -> float:
        """Sped up version of get_score that will be used if no aggregation, script or getter needs to be specified.

        Args:
            splits: The splits to consider.
            languages: The languages to consider. Can be ISO language codes or ISO language script codes.
            subsets: The hf_subsets to consider.

        Returns:
            The mean main score for the specified splits, languages and subsets.
        """
        if splits is None:
            splits = self.scores.keys()
        val_sum = 0
        n_val = 0
        for split in splits:
            if split not in self.scores:
                raise ValueError(f"Split missing from scores: {split}")

            for scores in self.scores[split]:
                langs = scores["languages"]
                hf_subset = scores["hf_subset"]
                main_score = scores.get("main_score", None)
                if main_score is None:
                    raise ValueError(f"Missing main score for subset: {hf_subset}")
                if subsets and hf_subset not in subsets:
                    continue
                elif subsets:
                    val_sum += main_score
                    n_val += 1
                    continue

                if languages is None:
                    val_sum += main_score
                    n_val += 1
                    continue
                for lang in langs:
                    if lang.split("-")[0] in languages:
                        val_sum += main_score
                        n_val += 1
                        logger.info(f"{val_sum=}, {n_val=}")
                        break
        if n_val == 0:
            raise ValueError("No splits had scores for the specified languages.")
        return val_sum / n_val

    @classmethod
    def from_validated(cls, **data) -> TaskResult:
        """Create a TaskResult from validated data.

        Returns:
            The created TaskResult object.
        """
        return cls.model_construct(**data)

    def __repr__(self) -> str:
        return f"TaskResult(task_name={self.task_name}, scores=...)"

    def only_main_score(self) -> TaskResult:
        """Return a new TaskResult object with only the main score.

        Returns:
            A new TaskResult object with only the main score.
        """
        new_scores: dict[str, list[Score]] = {}
        for split in self.scores:
            new_scores[split] = []
            for subset_scores in self.scores[split]:
                new_scores[split].append(
                    {
                        "hf_subset": subset_scores.get("hf_subset", "default"),
                        "main_score": subset_scores.get("main_score", np.nan),
                        "languages": subset_scores.get("languages", []),
                    }
                )
        new_res = {**self.to_dict(), "scores": new_scores}
        return TaskResult.from_validated(**new_res)

    def validate_and_filter_scores(self, task: AbsTask | None = None) -> TaskResult:
        """Validate and filter the scores against the task metadata.

        This ensures that the scores are correct for the given task, by removing any splits besides those specified in the task metadata.
        Additionally it also ensure that all of the splits required as well as the languages are present in the scores.
        Returns new TaskResult object.

        Args:
            task: The task to validate the scores against. E.g. if the task supplied is limited to certain splits and languages,
                the scores will be filtered to only include those splits and languages. If None it will attempt to get the task from the task_name.

        Returns:
            A new TaskResult object with the validated and filtered scores.
        """
        from mteb.get_tasks import get_task

        if task is None:
            task = get_task(self.task_name)

        splits = task.eval_splits
        hf_subsets = set(task.hf_subsets)  # Convert to set once

        new_scores: dict[str, list[Score]] = {}
        seen_splits = set()
        for split in self.scores:
            if split not in splits:
                continue
            seen_subsets = set()
            # Use list comprehension for better performance
            new_scores[split] = [
                _scores
                for _scores in self.scores[split]
                if _scores["hf_subset"] in hf_subsets
            ]
            for _scores in new_scores[split]:
                seen_subsets.add(_scores["hf_subset"])

            if seen_subsets != hf_subsets:
                missing_subsets = hf_subsets - seen_subsets
                if len(missing_subsets) > 2:
                    subset1, subset2 = list(missing_subsets)[:2]
                    missing_subsets_str = f"{{'{subset1}', '{subset2}', ...}}"
                else:
                    missing_subsets_str = str(missing_subsets)

                msg = f"{task.metadata.name}: Missing subsets {missing_subsets_str} for split {split}"
                logger.warning(msg)
                warnings.warn(msg)
            seen_splits.add(split)
        if seen_splits != set(splits):
            msg = f"{task.metadata.name}: Missing splits {set(splits) - seen_splits}"
            logger.warning(msg)
            warnings.warn(msg)
        data = self.model_dump()
        data["scores"] = new_scores
        return type(self).model_construct(**data)

    def is_mergeable(
        self,
        result: TaskResult | AbsTask,
        criteria: list[str] | list[Criteria] = [
            "mteb_version",
            "dataset_revision",
        ],
        raise_error: bool = False,
    ) -> bool:
        """Checks if the TaskResult object can be merged with another TaskResult or Task.

        Args:
            result: The TaskResult or Task object to check against.
            criteria: Additional criteria to check for merging. Can be "mteb_version" or "dataset_revision".
                It will always check that the task name match.
            raise_error: If True, raises an error if the objects cannot be merged. If False, returns False.

        Returns:
            True if the TaskResult object can be merged with the other object, False otherwise.
        """
        criteria = [Criteria.from_str(c) if isinstance(c, str) else c for c in criteria]
        if isinstance(result, TaskResult):
            name = result.task_name
            revision = result.dataset_revision
            mteb_version = result.mteb_version
        elif isinstance(result, AbsTask):
            mteb_version = version("mteb")
            name = result.metadata.name
            revision = result.metadata.revision
        else:
            msg = "result must be a TaskResult or AbsTask object"
            if raise_error:
                raise ValueError(msg)
            logger.debug(msg)
            return False

        if self.task_name != name:
            msg = f"Cannot merge TaskResult objects as they are derived from different tasks ({self.task_name} and {name})"
            if raise_error:
                raise ValueError(msg)
            logger.debug(msg)
            return False

        if Criteria.MTEB_VERSION in criteria and self.mteb_version != mteb_version:
            msg = f"Cannot merge TaskResult objects as they are derived from different MTEB versions ({self.mteb_version} (loaded) and {mteb_version} (current))"
            if raise_error:
                raise ValueError(msg)
            logger.debug(msg)
            return False

        if Criteria.DATASET_REVISION in criteria and self.dataset_revision != revision:
            msg = f"Cannot merge TaskResult objects as they are derived from different dataset revisions ({self.dataset_revision} and {revision})"
            if raise_error:
                raise ValueError(msg)
            logger.debug(msg)
            return False

        return True

    def merge(
        self,
        new_results: TaskResult,
        criteria: list[str] | list[Criteria] = [
            "mteb_version",
            "dataset_revision",
        ],
    ) -> TaskResult:
        """Merges two TaskResult objects.

        Args:
            new_results: The new TaskResult object to merge with the current one.
            criteria: Additional criteria to check for merging. Can be "mteb_version" or "dataset_revision".
                It will always check that the task name match.

        Returns:
            A new TaskResult object with the merged scores.
        """
        self.is_mergeable(new_results, criteria=criteria, raise_error=True)

        merged_scores = self.scores.copy()

        for split, scores in new_results.scores.items():
            if split in merged_scores:
                merged_scores[split] = self._merge_split_scores(
                    merged_scores[split], scores
                )
            else:
                merged_scores[split] = scores

        existing_kg_co2_emissions = (
            self.kg_co2_emissions if self.kg_co2_emissions else 0
        )
        new_kg_co2_emissions = (
            new_results.kg_co2_emissions if new_results.kg_co2_emissions else 0
        )
        merged_kg_co2_emissions = None
        if existing_kg_co2_emissions and new_kg_co2_emissions:
            merged_kg_co2_emissions = existing_kg_co2_emissions + new_kg_co2_emissions

        merged_evaluation_time = None
        if self.evaluation_time and new_results.evaluation_time:
            merged_evaluation_time = self.evaluation_time + new_results.evaluation_time
        merged_results = TaskResult(
            dataset_revision=new_results.dataset_revision,
            task_name=new_results.task_name,
            mteb_version=new_results.mteb_version,
            scores=merged_scores,
            evaluation_time=merged_evaluation_time,
            kg_co2_emissions=merged_kg_co2_emissions,
        )

        return merged_results

    @staticmethod
    def _merge_split_scores(
        existing_scores: list[ScoresDict], new_scores: list[ScoresDict]
    ) -> list[ScoresDict]:
        merged = {score["hf_subset"]: score for score in existing_scores}
        for score in new_scores:
            merged[score["hf_subset"]] = score
        return list(merged.values())

    def get_missing_evaluations(self, task: AbsTask) -> dict[str, list[str]]:
        """Checks which splits and subsets are missing from the results.

        Args:
            task: The task to check against.

        Returns:
            A dictionary with the splits as keys and a list of missing subsets as values.
        """
        missing_splits = {}
        for splits in task.eval_splits:
            if splits not in self.scores:  # split it fully missing
                missing_splits[splits] = task.hf_subsets
            if splits in self.scores:
                hf_subsets = {score["hf_subset"] for score in self.scores[splits]}
                missing_subsets = list(set(task.hf_subsets) - hf_subsets)
                if missing_subsets:
                    missing_splits[splits] = missing_subsets

        return missing_splits

    def get_hf_eval_results(self) -> list[EvalResult]:
        """Create HF evaluation results objects from TaskResult objects.

        Returns:
            List of EvalResult objects for each split and subset.
        """
        task_metadata = self.task.metadata
        task_type = task_metadata._hf_task_type()[0]
        results = []
        for split, scores in self.scores.items():
            for subset_results in scores:
                subset = subset_results.get("hf_subset", "default")
                results.append(
                    EvalResult(
                        task_type=task_type,
                        task_name=task_metadata.type,
                        dataset_type=task_metadata.dataset["path"],
                        dataset_name=f"{task_metadata.name} ({subset})",
                        dataset_config=subset,
                        dataset_split=split,
                        dataset_revision=task_metadata.dataset["revision"],
                        metric_type=task_metadata.main_score,
                        metric_name=task_metadata.main_score,
                        metric_value=subset_results["main_score"],
                        source_name="MTEB",
                        source_url="https://github.com/embeddings-benchmark/mteb/",
                    )
                )
        return results


class TaskError(BaseModel):
    """A class to represent an error that occurred during the evaluation of a task.

    Attributes:
        task_name: The name of the MTEB task.
        exception: The error message that occurred during the evaluation.
    """

    task_name: str
    exception: str
