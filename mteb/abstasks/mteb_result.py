from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from pydantic import BaseModel, field_validator

from mteb.abstasks.languages import LanguageScripts
from mteb.abstasks.TaskMetadata import (
    ISO_LANGUAGE,
    ISO_LANGUAGE_SCRIPT,
    LANGUAGES,
    HFSubset,
)
from mteb.overview import TASKS_REGISTRY

Split = str
Scores = Dict[str, Any]


def eval_langs_as_dict(
    eval_langs: LANGUAGES,
) -> dict[HFSubset, list[ISO_LANGUAGE_SCRIPT]]:
    if isinstance(eval_langs, list):
        return {"": eval_langs}
    return eval_langs  # type: ignore


class MTEBResults(BaseModel):
    """A class to represent the MTEB result.

    Attributes:
        task_name: The name of the MTEB task.
        dataset_revision: The revision dataset for the task on HuggingFace dataset hub.
        mteb_version: The version of the MTEB used to evaluate the model.
        scores: The scores of the model on the dataset. The scores is a dictionary with the following structure; dict[Split, dict[HFSubset, Scores]].
            Where Scores is a dictionary with the following structure; dict[str, Any]. Where the keys and values are scores. Split is the split of
            the dataset. HFSubset is the subset name on huggingface e.g. "en-de". HFSubset is set to "" if there is no subsets on huggingface.

    Example:
        >>> scores = {
        ...     "train": {
        ...         "en-de": {
        ...             "main_score": 0.5,
        ...             "evaluation_time": 100,
        ...         },
        ...         "en-fr": {
        ...             "main_score": 0.6,
        ...             "evaluation_time": 200,
        ...         },
        ...     },
        ... }
        >>> mteb_results = MTEBResults(
        ...     dataset_revision="1.0",
        ...     task_name="sample_task",
        ...     mteb_version="1.0",
        ...     scores=scores,
        ... )
        >>> mteb_results.get_main_score()  # get the main score for all languages
        0.55
        >>> mteb_results.get_main_score(languages=["fra"])  # get the main score for French
        0.6
    """

    dataset_revision: str
    task_name: str
    mteb_version: str
    scores: dict[Split, dict[HFSubset, Scores]]

    @field_validator("scores")
    def _validate_scores(
        cls, v: dict[Split, dict[HFSubset, Scores]]
    ) -> dict[Split, dict[HFSubset, Scores]]:
        for split, hf_subset_scores in v.items():
            for lang, scores in hf_subset_scores.items():
                cls._validate_scores_dict(scores)
        return v

    @staticmethod
    def _validate_scores_dict(scores: Scores) -> None:
        if "evaluation_time" not in scores:
            raise ValueError("'evaluation_time' should be in scores")

        if "main_score" not in scores:
            raise ValueError("'main_score' should be in scores")

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
        getter: Callable[[Scores], float],
        aggregation: Callable[[list[float]], Any],
        splits: list[Split] | None = None,
        languages: list[ISO_LANGUAGE | ISO_LANGUAGE_SCRIPT] | None = None,
        scripts: list[ISO_LANGUAGE_SCRIPT] | None = None,
    ) -> Any:
        """Get a score for the specified splits, languages, scripts and aggregation function.

        Args:
            getter: A function that takes a scores dictionary and returns a score e.g. "main_score" or "evaluation_time".
            aggregation: The aggregation function to use.
            splits: The splits to consider.
            languages: The languages to consider. Can be ISO language codes or ISO language script codes.
            scripts: The scripts to consider.

        Returns:
            The result of the aggregation function on the scores.
        """
        meta = TASKS_REGISTRY[self.task_name].metadata
        hf_subset2langs = eval_langs_as_dict(meta.eval_langs)

        if splits is None:
            splits = list(self.scores.keys())

        lang_scripts = LanguageScripts.from_languages_and_scripts(languages, scripts)

        values = []
        for split in splits:
            if split not in self.scores:
                raise ValueError(f"Split {split} not found in scores")

            for hf_subset, scores in self.scores[split].items():
                eval_langs = hf_subset2langs[hf_subset]
                include_subset = False
                for lang in eval_langs:
                    if lang_scripts.contains_language(lang):
                        include_subset = True
                        break
                if include_subset:
                    values.append(getter(scores))

            return aggregation(values)

    def get_main_score(
        self,
        splits: list[Split] | None = None,
        languages: list[ISO_LANGUAGE | ISO_LANGUAGE_SCRIPT] | None = None,
        scripts: list[ISO_LANGUAGE_SCRIPT] | None = None,
        aggregation: Callable[[list[float]], float] = np.mean,
    ) -> float:
        """Get the main score for the specified splits, languages, scripts and aggregation function.

        Args:
            splits: The splits to consider.
            languages: The languages to consider. Can be ISO language codes or ISO language script codes.
            scripts: The scripts to consider.
            aggregation: The aggregation function to use.
        """
        return self.get_score(
            getter=lambda scores: scores["main_score"],
            aggregation=aggregation,
            splits=splits,
            languages=languages,
            scripts=scripts,
        )

    def __repr__(self) -> str:
        return f"MTEBResults(task_name={self.task_name}, scores=...)"
