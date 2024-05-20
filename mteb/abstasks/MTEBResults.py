from __future__ import annotations

import json
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from pydantic import BaseModel, field_validator

from mteb.abstasks import AbsTask
from mteb.abstasks.languages import LanguageScripts
from mteb.abstasks.TaskMetadata import (
    ISO_LANGUAGE,
    ISO_LANGUAGE_SCRIPT,
    HFSubset,
)

Split = str
ScoresDict = Dict[str, Any]
# ^ e.g {'main_score': 0.5, 'hf_subset': 'en-de', 'languages': ['eng-Latn', 'deu-Latn']}
Score = Any


class MTEBResults(BaseModel):
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
        >>> mteb_results = MTEBResults.from_task_results(sample_task, scores)
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
    mteb_version: str
    scores: dict[Split, list[ScoresDict]]
    evaluation_time: float
    kg_co2_emissions: float | None = None

    @classmethod
    def from_task_results(
        cls,
        task: AbsTask,
        scores: dict[Split, dict[HFSubset, ScoresDict]],
        evaluation_time: float,
        kg_co2_emissions: float | None = None,
    ) -> MTEBResults:
        task_meta = task.metadata
        subset2langscripts = task_meta.hf_subsets_to_langscripts
        flat_scores = defaultdict(list)
        for split, hf_subset_scores in scores.items():
            for hf_subset, scores in hf_subset_scores.items():
                eval_langs = subset2langscripts[hf_subset]
                _scores = {
                    **scores,
                    "hf_subset": hf_subset,
                    "languages": eval_langs,
                }
                flat_scores[split].append(_scores)

        return MTEBResults(
            dataset_revision=task.metadata.dataset["revision"],
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

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict) -> MTEBResults:
        return cls.model_validate(data)

    def to_disk(self, path: Path) -> None:
        with path.open("w") as f:
            f.write(self.model_dump_json())

    @classmethod
    def from_disk(cls, path: Path) -> MTEBResults:
        with path.open("r") as f:
            return cls.model_validate(json.load(f))

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

    def __repr__(self) -> str:
        return f"MTEBResults(task_name={self.task_name}, scores=...)"
