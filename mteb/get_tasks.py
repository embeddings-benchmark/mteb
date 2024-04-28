"""This script contains functions that are used to get an overview of the MTEB benchmark."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Dict

from mteb.abstasks import AbsTask
from mteb.abstasks.languages import (
    ISO_TO_LANGUAGE,
    ISO_TO_SCRIPT,
    path_to_lang_codes,
    path_to_lang_scripts,
)
from mteb.abstasks.TaskMetadata import TASK_CATEGORY, TASK_DOMAIN, TASK_TYPE
from mteb.tasks import *  # import all tasks

logger = logging.getLogger(__name__)


def check_is_valid_script(script: str) -> None:
    if script not in ISO_TO_SCRIPT:
        raise ValueError(
            f"Invalid script code: {script}, you can find valid ISO 15924 codes in {path_to_lang_scripts}"
        )


def check_is_valid_language(lang: str) -> None:
    if lang not in ISO_TO_LANGUAGE:
        raise ValueError(
            f"Invalid language code: {lang}, you can find valid ISO 639-3 codes in {path_to_lang_codes}"
        )


def filter_superseeded_datasets(tasks: list[AbsTask]) -> list[AbsTask]:
    return [t for t in tasks if t.superseeded_by is None]


def filter_tasks_by_languages(
    tasks: list[AbsTask], languages: list[str]
) -> list[AbsTask]:
    [check_is_valid_language(lang) for lang in languages]
    langs_to_keep = set(languages)
    return [t for t in tasks if langs_to_keep.intersection(t.metadata.languages)]


def filter_tasks_by_script(tasks: list[AbsTask], script: list[str]) -> list[AbsTask]:
    [check_is_valid_script(s) for s in script]
    script_to_keep = set(script)
    return [t for t in tasks if script_to_keep.intersection(t.metadata.scripts)]


def filter_tasks_by_domains(
    tasks: list[AbsTask], domains: list[TASK_DOMAIN]
) -> list[AbsTask]:
    domains_to_keep = set(domains)

    def _convert_to_set(domain: list[TASK_DOMAIN] | None) -> set:
        return set(domain) if domain is not None else set()

    return [
        t
        for t in tasks
        if domains_to_keep.intersection(_convert_to_set(t.metadata.domains))
    ]


def filter_tasks_by_task_types(
    tasks: list[AbsTask], task_types: list[TASK_TYPE]
) -> list[AbsTask]:
    _task_types = set(task_types)
    return [t for t in tasks if t.metadata.type in _task_types]


def filter_task_by_categories(
    tasks: list[AbsTask], categories: list[TASK_CATEGORY]
) -> list[AbsTask]:
    _categories = set(categories)
    return [t for t in tasks if t.metadata.category in _categories]


def get_tasks(
    languages: list[str] | None = None,
    script: list[str] | None = None,
    domains: list[TASK_DOMAIN] | None = None,
    task_types: list[TASK_TYPE] | None = None,
    categories: list[TASK_CATEGORY] | None = None,
    exclude_superseeded: bool = True,
) -> list[AbsTask]:
    """Get a list of tasks based on the specified filters.

    Args:
        languages: A list of languages either specified as 3 letter languages codes (ISO 639-3, e.g. "eng") or as script languages codes e.g.
            "eng-Latn".
        script: A list of script codes (ISO 15924 codes). If None, all scripts are included.
        domains: A list of task domains.
        task_types: A string specifying the type of task. If None, all tasks are included.
        categories: A list of task categories these include "s2s" (sentence to sentence), "s2p" (sentence to paragraph) and "p2p" (paragraph to
            paragraph).
        exclude_superseeded: A boolean flag to exclude datasets which are superseeded by another.

    Returns:
        A list of all initialized tasks objects which pass all of the filters (AND operation).

    Examples:
        >>> get_tasks(languages=["eng", "deu"], script=["Latn"], domains=["Legal"])
        >>> get_tasks(languages=["eng"], script=["Latn"], task_types=["Classification"])
        >>> get_tasks(languages=["eng"], script=["Latn"], task_types=["Clustering"], exclude_superseeded=False)
    """
    tasks_categories_cls = [cls for cls in AbsTask.__subclasses__()]
    tasks = [
        cls()
        for cat_cls in tasks_categories_cls
        for cls in cat_cls.__subclasses__()
        if cat_cls.__name__.startswith("AbsTask")
    ]

    if languages:
        tasks = filter_tasks_by_languages(tasks, languages)
    if script:
        tasks = filter_tasks_by_script(tasks, script)
    if domains:
        tasks = filter_tasks_by_domains(tasks, domains)
    if task_types:
        tasks = filter_tasks_by_task_types(tasks, task_types)
    if categories:
        tasks = filter_task_by_categories(tasks, categories)
    if exclude_superseeded:
        tasks = filter_superseeded_datasets(tasks)

    return tasks


def count_languages(
    domains: list[TASK_DOMAIN] | None = None,
    task_types: list[TASK_TYPE] | None = None,
    categories: list[TASK_CATEGORY] | None = None,
    exclude_superseeded: bool = True,
) -> Dict:
    """Get a summary of language count.

    Args:
        domains: A list of task domains.
        task_types: A string specifying the type of task. If None, all tasks are included.
        categories: A list of task categories these include "s2s" (sentence to sentence), "s2p" (sentence to paragraph) and "p2p" (paragraph to
            paragraph).
        exclude_superseeded: A boolean flag to exclude datasets which are superseeded by another.

    Returns:
        A dictionary with language occurence, per domain, task type, or category.

    Examples:
        >>> count_languages(task_types=['Clustering'])
    """
    tasks = get_tasks(
        domains=domains,
        task_types=task_types,
        categories=categories,
        exclude_superseeded=exclude_superseeded,
    )
    langs = []
    for task in tasks:
        langs.extend(task.languages)
    return Counter(langs)
