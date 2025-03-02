from __future__ import annotations

import json
import logging
from argparse import Namespace
from collections import defaultdict
from collections.abc import Iterable
from functools import cached_property
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

import numpy as np
from packaging.version import Version
from pydantic import BaseModel, field_validator

from mteb.abstasks.AbsTask import AbsTask, ScoresDict
from mteb.abstasks.TaskMetadata import ISO_LANGUAGE_SCRIPT, HFSubset
from mteb.languages import ISO_LANGUAGE, LanguageScripts

Split = str
Score = Any

logger = logging.getLogger(__name__)


class ScalaNbClassificationDummy:
    """A dummy task for loading historic results from before v1.11.0"""

    metadata = Namespace(  # type: ignore
        name="ScalaNbClassification",
        main_score="accuracy",
        type="Classification",
        hf_subsets_to_langscripts={
            "default": ["nob-Latn"],
        },
        dataset={"revision": "revision_not_applicable"},
        revision="revision_not_applicable",
    )


class ScalaNnClassificationDummy:
    """A dummy task for loading historic results from before v1.11.0"""

    metadata = Namespace(  # type: ignore
        name="ScalaNnClassification",
        main_score="accuracy",
        type="Classification",
        hf_subsets_to_langscripts={
            "default": ["nno-Latn"],
        },
        dataset={"revision": "revision_not_applicable"},
        revision="revision_not_applicable",
    )


class ScalaDaClassificationDummy:
    """A dummy task for loading historic results from before v1.11.0"""

    metadata = Namespace(  # type: ignore
        name="ScalaDaClassification",
        main_score="accuracy",
        type="Classification",
        hf_subsets_to_langscripts={
            "default": ["dan-Latn"],
        },
        dataset={"revision": "revision_not_applicable"},
        revision="revision_not_applicable",
    )


class ScalaSvClassificationDummy:
    """A dummy task for loading historic results from before v1.11.0"""

    metadata = Namespace(  # type: ignore
        name="ScalaSvClassification",
        main_score="accuracy",
        type="Classification",
        hf_subsets_to_langscripts={
            "default": ["swe-Latn"],
        },
        dataset={"revision": "revision_not_applicable"},
        revision="revision_not_applicable",
    )


outdated_tasks = {
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
        scores: The scores of the model on the dataset. The scores is a dictionary with the following structure; dict[Split, list[Scores]].
            Where Scores is a dictionary with the following structure; dict[str, Any]. Where the keys and values are scores. Split is the split of
            the dataset.
        evaluation_time: The time taken to evaluate the model.
        kg_co2_emissions: The kg of CO2 emissions produced by the model during evaluation.

    Example:
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
    scores: dict[Split, list[ScoresDict]]
    evaluation_time: float | None
    kg_co2_emissions: float | None = None

    @classmethod
    def from_task_results(
        cls,
        task: AbsTask | type[AbsTask],
        scores: dict[Split, dict[HFSubset, ScoresDict]],
        evaluation_time: float,
        kg_co2_emissions: float | None = None,
    ) -> TaskResult:
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
    def _validate_scores(
        cls, v: dict[Split, list[ScoresDict]]
    ) -> dict[Split, list[ScoresDict]]:
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
        langs = []
        for split, split_res in self.scores.items():
            for entry in split_res:
                langs.extend([lang.split("-")[0] for lang in entry["languages"]])
        return list(set(langs))

    @cached_property
    def task(self) -> AbsTask:
        from mteb.overview import get_task

        return get_task(self.task_name)

    @property
    def domains(self) -> list[str]:
        doms = self.task.metadata.domains
        if doms is None:
            doms = []
        return doms

    @property
    def task_type(self) -> str:
        return self.task.metadata.type

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> TaskResult:
        return cls.model_validate(data)

    def _round_scores(self, scores: dict[Split, list[ScoresDict]], n: int) -> None:
        """Recursively round scores to n decimal places"""
        for key, value in scores.items():
            if isinstance(value, dict):
                self._round_scores(value, n)
            elif isinstance(value, list):
                for i, v in enumerate(value):
                    if isinstance(v, dict):
                        self._round_scores(v, n)
                    elif isinstance(v, float):
                        value[i] = round(v, n)

            elif isinstance(value, float):
                scores[key] = round(value, n)

    def to_disk(self, path: Path) -> None:
        json_obj = self.model_dump()
        self._round_scores(json_obj["scores"], 6)

        with path.open("w") as f:
            json.dump(json_obj, f, indent=2)

    @classmethod
    def from_disk(cls, path: Path, load_historic_data: bool = True) -> TaskResult:  # type: ignore
        """Load TaskResult from disk.

        Args:
            path: The path to the file to load.
            load_historic_data: Whether to attempt to load historic data from before v1.11.0.
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
            obj = cls.model_validate(data)
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
                                hf_subset_scores[f"{key}_{k}"] = v
                            hf_subset_scores.pop(key)

    @classmethod
    def _convert_from_before_v1_11_0(cls, data: dict) -> TaskResult:
        from mteb.overview import TASKS_REGISTRY

        # in case the task name is not found in the registry, try to find a lower case version
        lower_case_registry = {k.lower(): v for k, v in TASKS_REGISTRY.items()}

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
            task = TASKS_REGISTRY.get(task_name, lower_case_registry[task_name.lower()])

        # make sure that main score exists
        main_score = task.metadata.main_score
        for split, split_score in scores.items():
            for hf_subset, hf_subset_scores in split_score.items():
                for name, prev_name in [
                    ("cosine", "cos_sim"),
                    ("manhattan", "manhattan"),
                    ("euclidean", "euclidean"),
                    ("dot", "dot"),
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
                        logger.warning(f"Main score {main_score} not found in scores")
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
            task,  # type: ignore
            scores,
            evaluation_time,
            kg_co2_emissions=None,
        )
        result.dataset_revision = dataset_revision
        result.mteb_version = mteb_version
        return result

    def get_score(
        self,
        splits: list[Split] | None = None,
        languages: list[ISO_LANGUAGE | ISO_LANGUAGE_SCRIPT] | None = None,
        scripts: list[ISO_LANGUAGE_SCRIPT] | None = None,
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

    def get_score_fast(
        self,
        splits: Iterable[str] | None = None,
        languages: str | None = None,
        subsets: Iterable[str] | None = None,
    ) -> float:
        """Sped up version of get_score that will be used if no aggregation, script or getter needs to be specified."""
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
        return cls.model_construct(**data)

    def __repr__(self) -> str:
        return f"TaskResult(task_name={self.task_name}, scores=...)"

    def only_main_score(self) -> TaskResult:
        new_scores = {}
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
        new_res = TaskResult.from_validated(**new_res)
        return new_res

    def validate_and_filter_scores(self, task: AbsTask | None = None) -> TaskResult:
        """This ensures that the scores are correct for the given task, by removing any splits besides those specified in the task metadata.
        Additionally it also ensure that all of the splits required as well as the languages are present in the scores.
        Returns new TaskResult object.

        Args:
            task: The task to validate the scores against. E.g. if the task supplied is limited to certain splits and languages,
                the scores will be filtered to only include those splits and languages. If None it will attempt to get the task from the task_name.
        """
        from mteb.overview import get_task

        if task is None:
            task = get_task(self.task_name)

        splits = task.eval_splits
        hf_subsets = task.hf_subsets
        hf_subsets = set(hf_subsets)

        new_scores = {}
        seen_splits = set()
        for split in self.scores:
            if split not in splits:
                continue
            new_scores[split] = []
            seen_subsets = set()
            for _scores in self.scores[split]:
                if _scores["hf_subset"] not in hf_subsets:
                    continue
                new_scores[split].append(_scores)
                seen_subsets.add(_scores["hf_subset"])
            if seen_subsets != hf_subsets:
                missing_subsets = hf_subsets - seen_subsets
                if len(missing_subsets) > 2:
                    subset1, subset2 = list(missing_subsets)[:2]
                    missing_subsets_str = f"{{'{subset1}', '{subset2}', ...}}"
                else:
                    missing_subsets_str = str(missing_subsets)

                logger.warning(
                    f"{task.metadata.name}: Missing subsets {missing_subsets_str} for split {split}"
                )
            seen_splits.add(split)
        if seen_splits != set(splits):
            logger.warning(
                f"{task.metadata.name}: Missing splits {set(splits) - seen_splits}"
            )
        new_res = {**self.to_dict(), "scores": new_scores}
        new_res = TaskResult.from_validated(**new_res)
        return new_res
