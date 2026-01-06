"""This script contains functions that are used to get an overview of the MTEB benchmark."""

import logging
from collections.abc import Iterable, Sequence
from typing import overload

from mteb.abstasks import (
    AbsTask,
)
from mteb.abstasks.aggregated_task import AbsTaskAggregate
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
    tasks: Iterable[AbsTask],
    *,
    languages: Sequence[str] | None = None,
    script: Sequence[str] | None = None,
    domains: Iterable[TaskDomain] | None = None,
    task_types: Iterable[TaskType] | None = None,
    categories: Iterable[TaskCategory] | None = None,
    modalities: Iterable[Modalities] | None = None,
    exclusive_modality_filter: bool = False,
    exclude_superseded: bool = False,
    exclude_aggregate: bool = False,
    exclude_private: bool = False,
) -> list[AbsTask]: ...


@overload
def filter_tasks(
    tasks: Iterable[type[AbsTask]],
    *,
    languages: Sequence[str] | None = None,
    script: Sequence[str] | None = None,
    domains: Iterable[TaskDomain] | None = None,
    task_types: Iterable[TaskType] | None = None,
    categories: Iterable[TaskCategory] | None = None,
    modalities: Iterable[Modalities] | None = None,
    exclusive_modality_filter: bool = False,
    exclude_superseded: bool = False,
    exclude_aggregate: bool = False,
    exclude_private: bool = False,
) -> list[type[AbsTask]]: ...


def filter_tasks(
    tasks: Iterable[AbsTask] | Iterable[type[AbsTask]],
    *,
    languages: Sequence[str] | None = None,
    script: Sequence[str] | None = None,
    domains: Iterable[TaskDomain] | None = None,
    task_types: Iterable[TaskType] | None = None,
    categories: Iterable[TaskCategory] | None = None,
    modalities: Iterable[Modalities] | None = None,
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
        modalities: A list of modalities to include. If None, all modalities are included.
        exclusive_modality_filter: If True, only keep tasks where _all_ filter modalities are included in the
            task's modalities and ALL task modalities are in filter modalities (exact match).
            If False, keep tasks if _any_ of the task's modalities match the filter modalities.
        exclude_aggregate: If True, exclude aggregate tasks. If False, both aggregate and non-aggregate tasks are returned.
        exclude_private: If True (default), exclude private/closed datasets (is_public=False). If False, include both public and private datasets.

    Returns:
        A list of tasks objects which pass all of the filters.

    Examples:
        >>> text_classification_tasks = filter_tasks(my_tasks, task_types=["Classification"], modalities=["text"])
        >>> medical_tasks = filter_tasks(my_tasks, domains=["Medical"])
        >>> english_tasks = filter_tasks(my_tasks, languages=["eng"])
        >>> latin_script_tasks = filter_tasks(my_tasks, script=["Latn"])
        >>> text_image_tasks = filter_tasks(my_tasks, modalities=["text", "image"], exclusive_modality_filter=True)

    """
    langs_to_keep = None
    if languages:
        [_check_is_valid_language(lang) for lang in languages]  # type: ignore[func-returns-value]
        langs_to_keep = set(languages)

    script_to_keep = None
    if script:
        [_check_is_valid_script(s) for s in script]  # type: ignore[func-returns-value]
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
        # For metadata and superseded_by, we can access them directly
        metadata = t.metadata

        if langs_to_keep and not langs_to_keep.intersection(metadata.languages):
            continue
        if script_to_keep and not script_to_keep.intersection(metadata.scripts):
            continue
        if domains_to_keep and not domains_to_keep.intersection(
            _convert_to_set(metadata.domains)
        ):
            continue
        if task_types_to_keep and metadata.type not in task_types_to_keep:
            continue
        if categories_to_keep and metadata.category not in categories_to_keep:
            continue
        if modalities_to_keep:
            if exclusive_modality_filter:
                if set(metadata.modalities) != modalities_to_keep:
                    continue
            else:
                if not modalities_to_keep.intersection(metadata.modalities):
                    continue
        if exclude_superseded and metadata.superseded_by is not None:
            continue
        is_aggregate = (
            issubclass(t, AbsTaskAggregate)
            if isinstance(t, type)
            else isinstance(t, AbsTaskAggregate)
        )
        if exclude_aggregate and is_aggregate:
            continue
        if exclude_private and not metadata.is_public:
            continue

        _tasks.append(t)

    return _tasks  # type: ignore[return-value]  # type checker cannot infer the overload return type
