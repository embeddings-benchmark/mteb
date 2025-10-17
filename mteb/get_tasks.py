"""This script contains functions that are used to get an overview of the MTEB benchmark."""

import difflib
import logging
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

import pandas as pd

from mteb.abstasks import (
    AbsTask,
)
from mteb.abstasks.task_metadata import TaskCategory, TaskDomain, TaskType
from mteb.filter_tasks import filter_tasks
from mteb.types import Modalities

logger = logging.getLogger(__name__)


# Create task registry
def _gather_tasks() -> tuple[type[AbsTask], ...]:
    import mteb.tasks as tasks

    tasks = [
        t
        for t in tasks.__dict__.values()
        if isinstance(t, type) and issubclass(t, AbsTask)
    ]
    return tuple(tasks)


def _create_name_to_task_mapping(
    tasks: Sequence[type[AbsTask]],
) -> dict[str, type[AbsTask]]:
    metadata_names = {}
    for cls in tasks:
        if cls.metadata.name in metadata_names:
            raise ValueError(
                f"Duplicate task name found: {cls.metadata.name}. Please make sure that all task names are unique."
            )
        metadata_names[cls.metadata.name] = cls
    return metadata_names


def _create_similar_tasks(tasks: Sequence[type[AbsTask]]) -> dict[str, list[str]]:
    """Create a dictionary of similar tasks.

    Returns:
        Dict with key is parent task and value is list of similar tasks.
    """
    similar_tasks = defaultdict(list)
    for task in tasks:
        if task.metadata.adapted_from:
            for similar_task in task.metadata.adapted_from:
                similar_tasks[similar_task].append(task.metadata.name)
        if task.metadata.superseded_by:
            similar_tasks[task.metadata.superseded_by].append(task.metadata.name)
    return similar_tasks


TASK_LIST = _gather_tasks()
_TASKS_REGISTRY = _create_name_to_task_mapping(TASK_LIST)
_SIMILAR_TASKS = _create_similar_tasks(TASK_LIST)


_DEFAULT_PROPRIETIES = (
    "name",
    "type",
    "languages",
    "domains",
    "license",
    "modalities",
)


class MTEBTasks(tuple[AbsTask]):
    """A tuple of tasks with additional methods to get an overview of the tasks."""

    def __repr__(self) -> str:
        return "MTEBTasks" + super().__repr__()

    @staticmethod
    def _extract_property_from_task(task: AbsTask, property: str):
        if hasattr(task.metadata, property):
            return getattr(task.metadata, property)
        elif hasattr(task, property):
            return getattr(task, property)
        else:
            raise KeyError("Property neither in Task attribute or in task metadata.")

    @property
    def languages(self) -> set:
        """Return all languages from tasks"""
        langs = set()
        for task in self:
            for lg in task.languages:
                langs.add(lg)
        return langs

    def count_languages(self) -> Counter:
        """Summarize count of all languages from tasks

        Returns:
            Counter with language as key and count as value.
        """
        langs = []
        for task in self:
            langs.extend(task.languages)
        return Counter(langs)

    def to_markdown(
        self,
        properties: Sequence[str] = _DEFAULT_PROPRIETIES,
        limit_n_entries: int | None = 3,
    ) -> str:
        """Generate markdown table with tasks summary

        Args:
            properties: list of metadata to summarize from a Task class.
            limit_n_entries: Limit the number of entries for cell values, e.g. number of languages and domains. Will use "..." to indicate that
                there are more entries.

        Returns:
            string with a markdown table.
        """

        def _limit_entries_in_cell_inner(cell: Any):
            if isinstance(cell, list | set):
                return self._limit_entries_in_cell(cell, limit_n_entries)
            return cell

        markdown_table = "| Task" + "".join([f"| {p}  " for p in properties]) + "|\n"
        _head_sep = "| ---" * (len(properties) + 1) + " |\n"
        markdown_table += _head_sep
        for task in self:
            markdown_table += f"| {task.metadata.name} "
            markdown_table += "".join(
                [
                    f"| {_limit_entries_in_cell_inner(self._extract_property_from_task(task, p))} "
                    for p in properties
                ]
            )
            markdown_table += " |\n"
        return markdown_table

    def to_dataframe(
        self,
        properties: Sequence[str] = _DEFAULT_PROPRIETIES,
    ) -> pd.DataFrame:
        """Generate pandas DataFrame with tasks summary

        Args:
            properties: list of metadata to summarize from a Task class.

        Returns:
            pandas DataFrame.
        """
        data = []
        for task in self:
            data.append(
                {p: self._extract_property_from_task(task, p) for p in properties}
            )
        return pd.DataFrame(data)

    @staticmethod
    def _limit_entries_in_cell(
        cell: list | set, limit_n_entries: int | None = 3
    ) -> str:
        if limit_n_entries and len(cell) > limit_n_entries:
            ending = "]" if isinstance(cell, list) else "}"
            cell = sorted(cell)
            return str(cell[:limit_n_entries])[:-1] + ", ..." + ending
        else:
            return str(cell)

    def to_latex(
        self,
        properties: Sequence[str] = _DEFAULT_PROPRIETIES,
        group_indices: Sequence[str] | None = ("type", "name"),
        include_citation_in_name: bool = True,
        limit_n_entries: int | None = 3,
    ) -> str:
        """Generate a LaTeX table of the tasks.

        Args:
            properties: list of metadata to summarize from a Task class.
            group_indices: list of properties to group the table by.
            include_citation_in_name: Whether to include the citation in the name.
            limit_n_entries: Limit the number of entries for cell values, e.g. number of languages and domains. Will use "..." to indicate that
                there are more entries.

        Returns:
            string with a LaTeX table.
        """
        if include_citation_in_name and "name" in properties:
            properties += ["intext_citation"]
            df = self.to_dataframe(properties)
            df["name"] = df["name"] + " " + df["intext_citation"]
            df = df.drop(columns=["intext_citation"])
        else:
            df = self.to_dataframe(properties)

        if limit_n_entries and df.shape[0]:  # ensure that there are entries
            for col in df.columns:
                # check if content is a list or set
                if isinstance(df[col].iloc[0], list | set):
                    _col = []
                    for val in df[col]:
                        str_col = self._limit_entries_in_cell(val, limit_n_entries)

                        # escape } and { characters
                        str_col = str_col.replace("{", "\\{").replace("}", "\\}")
                        _col.append(str_col)
                    df[col] = _col

        if group_indices:
            df = df.set_index(group_indices)

        return df.to_latex()


def get_tasks(
    tasks: list[str] | None = None,
    *,
    languages: list[str] | None = None,
    script: list[str] | None = None,
    domains: list[TaskDomain] | None = None,
    task_types: list[TaskType] | None = None,  # type: ignore
    categories: list[TaskCategory] | None = None,
    exclude_superseded: bool = True,
    eval_splits: list[str] | None = None,
    exclusive_language_filter: bool = False,
    modalities: list[Modalities] | None = None,
    exclusive_modality_filter: bool = False,
    exclude_aggregate: bool = False,
    exclude_private: bool = True,
) -> MTEBTasks:
    """Get a list of tasks based on the specified filters.

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
        exclusive_language_filter: Some datasets contains more than one language e.g. for STS22 the subset "de-en" contain eng and deu. If
            exclusive_language_filter is set to False both of these will be kept, but if set to True only those that contains all the languages
            specified will be kept.
        modalities: A list of modalities to include. If None, all modalities are included.
        exclusive_modality_filter: If True, only keep tasks where _all_ filter modalities are included in the
            task's modalities and ALL task modalities are in filter modalities (exact match).
            If False, keep tasks if _any_ of the task's modalities match the filter modalities.
        exclude_aggregate: If True, exclude aggregate tasks. If False, both aggregate and non-aggregate tasks are returned.
        exclude_private: If True (default), exclude private/closed datasets (is_public=False). If False, include both public and private datasets.

    Returns:
        A list of all initialized tasks objects which pass all of the filters (AND operation).

    Examples:
        >>> get_tasks(languages=["eng", "deu"], script=["Latn"], domains=["Legal"])
        >>> get_tasks(languages=["eng"], script=["Latn"], task_types=["Classification"])
        >>> get_tasks(languages=["eng"], script=["Latn"], task_types=["Clustering"], exclude_superseded=False)
        >>> get_tasks(languages=["eng"], tasks=["WikipediaRetrievalMultilingual"], eval_splits=["test"])
        >>> get_tasks(tasks=["STS22"], languages=["eng"], exclusive_language_filter=True) # don't include multilingual subsets containing English
    """
    if tasks:
        if domains or task_types or categories:
            logger.warning(
                "When `tasks` is provided, other filters like domains, task_types, and categories are ignored. "
                + "If you want to filter a list of tasks, please use `mteb.filter_tasks` instead."
            )
        _tasks = [
            get_task(
                task,
                languages,
                script,
                eval_splits=eval_splits,
                exclusive_language_filter=exclusive_language_filter,
            )
            for task in tasks
        ]
        return MTEBTasks(_tasks)

    _tasks = filter_tasks(
        TASK_LIST,
        languages=languages,
        script=script,
        domains=domains,
        task_types=task_types,
        categories=categories,
        modalities=modalities,
        exclusive_modality_filter=exclusive_modality_filter,
        exclude_superseded=exclude_superseded,
        exclude_aggregate=exclude_aggregate,
        exclude_private=exclude_private,
    )
    _tasks = [
        cls().filter_languages(languages, script).filter_eval_splits(eval_splits)
        for cls in _tasks
    ]

    return MTEBTasks(_tasks)


_TASK_RENAMES = {"PersianTextTone": "SynPerTextToneClassification"}


def get_task(
    task_name: str,
    languages: list[str] | None = None,
    script: list[str] | None = None,
    eval_splits: list[str] | None = None,
    hf_subsets: list[str] | None = None,
    exclusive_language_filter: bool = False,
) -> AbsTask:
    """Get a task by name.

    Args:
        task_name: The name of the task to fetch.
        languages: A list of languages either specified as 3 letter languages codes (ISO 639-3, e.g. "eng") or as script languages codes e.g.
            "eng-Latn". For multilingual tasks this will also remove languages that are not in the specified list.
        script: A list of script codes (ISO 15924 codes). If None, all scripts are included. For multilingual tasks this will also remove scripts
        eval_splits: A list of evaluation splits to include. If None, all splits are included.
        hf_subsets: A list of Huggingface subsets to evaluate on.
        exclusive_language_filter: Some datasets contains more than one language e.g. for STS22 the subset "de-en" contain eng and deu. If
            exclusive_language_filter is set to False both of these will be kept, but if set to True only those that contains all the languages
            specified will be kept.

    Returns:
        An initialized task object.

    Examples:
        >>> get_task("BornholmBitextMining")
    """
    if task_name in _TASK_RENAMES:
        _task_name = _TASK_RENAMES[task_name]
        logger.warning(
            f"The task with the given name '{task_name}' has been renamed to '{_task_name}'. To prevent this warning use the new name."
        )

    if task_name not in _TASKS_REGISTRY:
        close_matches = difflib.get_close_matches(task_name, _TASKS_REGISTRY.keys())
        if close_matches:
            suggestion = f"KeyError: '{task_name}' not found. Did you mean: '{close_matches[0]}'?"
        else:
            suggestion = (
                f"KeyError: '{task_name}' not found and no similar keys were found."
            )
        raise KeyError(suggestion)
    task = _TASKS_REGISTRY[task_name]()
    if eval_splits:
        task.filter_eval_splits(eval_splits=eval_splits)
    return task.filter_languages(
        languages,
        script,
        hf_subsets=hf_subsets,
        exclusive_language_filter=exclusive_language_filter,
    )
