"""This script contains functions that are used to get an overview of the MTEB benchmark."""

import logging
from collections.abc import Sequence
from typing import overload

from mteb.abstasks import (
    AbsTask,
)
from mteb.abstasks.task_metadata import TaskCategory, TaskDomain, TaskType
from mteb.languages import (
    ISO_TO_LANGUAGE,
    ISO_TO_SCRIPT,
)
from mteb.types import Modalities

logger = logging.getLogger(__name__)


def _check_is_valid_script(script: str) -> None:
    if script not in ISO_TO_SCRIPT:
        raise ValueError(
            f"Invalid script code: '{script}', you can see valid ISO 15924 codes using `from mteb.languages import ISO_TO_SCRIPT`."
        )


def _check_is_valid_language(lang: str) -> None:
    if lang not in ISO_TO_LANGUAGE:
        raise ValueError(
            f"Invalid language code: '{lang}', you can see valid ISO 639-3 codes using `from mteb.languages import ISO_TO_LANGUAGE`."
        )


@overload
def filter_tasks(
    tasks: Sequence[AbsTask],
    *,
    languages: list[str] | None = None,
    script: list[str] | None = None,
    domains: list[TaskDomain] | None = None,
    task_types: list[TaskType] | None = None,  # type: ignore
    categories: list[TaskCategory] | None = None,
    modalities: list[Modalities] | None = None,
    exclusive_modality_filter: bool = False,
    exclude_superseded: bool = False,
    exclude_aggregate: bool = False,
    exclude_private: bool = False,
) -> list[AbsTask]: ...


@overload
def filter_tasks(
    tasks: Sequence[type[AbsTask]],
    *,
    languages: list[str] | None = None,
    script: list[str] | None = None,
    domains: list[TaskDomain] | None = None,
    task_types: list[TaskType] | None = None,  # type: ignore
    categories: list[TaskCategory] | None = None,
    modalities: list[Modalities] | None = None,
    exclusive_modality_filter: bool = False,
    exclude_superseded: bool = False,
    exclude_aggregate: bool = False,
    exclude_private: bool = False,
) -> list[type[AbsTask]]: ...


def filter_tasks(
    tasks: Sequence[AbsTask] | Sequence[type[AbsTask]],
    *,
    languages: list[str] | None = None,
    script: list[str] | None = None,
    domains: list[TaskDomain] | None = None,
    task_types: list[TaskType] | None = None,  # type: ignore
    categories: list[TaskCategory] | None = None,
    modalities: list[Modalities] | None = None,
    exclusive_modality_filter: bool = False,
    exclude_superseded: bool = False,
    exclude_aggregate: bool = False,
    exclude_private: bool = False,
) -> list[AbsTask] | list[type[AbsTask]]:
    """Filter tasks based on the specified criteria.

    Args:
        tasks: A list of task names to include. If None, all tasks which pass the filters are included. If passed, other filters are ignored.
        languages: A list of languages either specified as 3 letter languages codes (ISO 639-3, e.g. "eng") or as script languages codes e.g.
            "eng-Latn". For multilingual tasks this will also remove languages that are not in the specified list.
        script: A list of script codes (ISO 15924 codes, e.g. "Latn"). If None, all scripts are included. For multilingual tasks this will also remove scripts
            that are not in the specified list.
        domains: A list of task domains, e.g. "Legal", "Medical", "Fiction".
        task_types: A string specifying the type of task e.g. "Classification" or "Retrieval". If None, all tasks are included.
        categories: A list of task categories these include "t2t" (text to text), "t2i" (text to image). See TaskMetadata for the full list.
        exclude_superseded: A boolean flag to exclude datasets which are superseded by another.
        eval_splits: A list of evaluation splits to include. If None, all splits are included.
        modalities: A list of modalities to include. If None, all modalities are included.
        exclusive_modality_filter: If True, only keep tasks where _all_ filter modalities are included in the
            task's modalities and ALL task modalities are in filter modalities (exact match).
            If False, keep tasks if _any_ of the task's modalities match the filter modalities.
        exclude_aggregate: If True, exclude aggregate tasks. If False, both aggregate and non-aggregate tasks are returned.
        exclude_private: If True (default), exclude private/closed datasets (is_public=False). If False, include both public and private datasets.

    Returns:
        A list of all initialized tasks objects which pass all of the filters (AND operation).

    """
    langs_to_keep = None
    if languages:
        [_check_is_valid_language(lang) for lang in languages]
        langs_to_keep = set(languages)

    script_to_keep = None
    if script:
        [_check_is_valid_script(s) for s in script]
        script_to_keep = set(script)

    domains_to_keep = None
    if domains:
        domains_to_keep = set(domains)

    def _convert_to_set(domain: list[TaskDomain] | None) -> set:
        return set(domain) if domain is not None else set()

    task_types_to_keep = None
    if task_types:
        task_types_to_keep = set(task_types)

    categories_to_keep = None
    if categories:
        categories_to_keep = set(categories)

    modalities_to_keep = None
    if modalities:
        modalities_to_keep = set(modalities)

    _tasks = []
    for t in tasks:
        if langs_to_keep and not langs_to_keep.intersection(t.metadata.languages):
            continue
        if script_to_keep and not script_to_keep.intersection(t.metadata.scripts):
            continue
        if domains_to_keep and not domains_to_keep.intersection(
            _convert_to_set(t.metadata.domains)
        ):
            continue
        if task_types_to_keep and t.metadata.type not in task_types_to_keep:
            continue
        if categories_to_keep and t.metadata.category not in categories_to_keep:
            continue
        if modalities_to_keep:
            if exclusive_modality_filter:
                if set(t.metadata.modalities) != modalities_to_keep:
                    continue
            else:
                if not modalities_to_keep.intersection(t.metadata.modalities):
                    continue
        if exclude_superseded and t.superseded_by is not None:
            continue
        if exclude_aggregate and t.is_aggregate:
            continue
        if exclude_private and not t.metadata.is_public:
            continue

        _tasks.append(t)

    return _tasks
