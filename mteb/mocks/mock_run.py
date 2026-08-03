"""Core logic for sanity checking a model implementation by running it on the mock tasks.

This module is free of any CLI specific logic (argument parsing, printing, writing files),
such that it can be used directly from Python:

```python
import mteb

model = mteb.get_model("sentence-transformers/all-MiniLM-L6-v2")
results = mteb.mock_run(model)
```

The `mteb mock-run` CLI command is a thin wrapper around the functions defined here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, get_args

from pydantic import BaseModel, ConfigDict

from mteb.evaluate import evaluate
from mteb.mocks import ALL_MOCK_TASK_TEST_GRID
from mteb.models.models_protocols import (
    CrossEncoderProtocol,
    EncoderProtocol,
    SearchProtocol,
)
from mteb.results.task_result import TaskError, TaskResult
from mteb.types import Modalities

if TYPE_CHECKING:
    from collections.abc import ItemsView

    from mteb.abstasks.abstask import AbsTask

logger = logging.getLogger(__name__)

MODALITIES: tuple[Modalities, ...] = get_args(Modalities)

# a task failing due to a missing optional dependency is reported as skipped and not as a failure
_DEPENDENCY_ERROR_PATTERNS = (
    "importerror",
    "modulenotfounderror",
    "dependencies",
    "install '",
)


class MockRunResults(BaseModel):
    """The results of running a model on the mock tasks, as returned by [`mteb.mock_run`][mteb.mock_run].

    Attributes:
        model_name: The name of the checked model.
        task_results: A mapping of every mock task name to its status; a `TaskResult` (the task
            ran successfully), a `TaskError` (the task failed) or `None` (the task is not
            compatible with the model).
    """

    model_config = ConfigDict(
        protected_namespaces=(),  # to free up the name model_name which is otherwise protected
    )

    model_name: str
    task_results: dict[str, TaskResult | TaskError | None]

    def __getitem__(self, task_name: str) -> TaskResult | TaskError | None:
        return self.task_results[task_name]

    def __contains__(self, task_name: str) -> bool:
        return task_name in self.task_results

    def items(self) -> ItemsView[str, TaskResult | TaskError | None]:
        """The (task name, status) pairs of every mock task."""
        return self.task_results.items()

    @property
    def all_passed(self) -> bool:
        """Whether all compatible mock tasks either passed or were skipped due to missing dependencies."""
        return all(
            not isinstance(status, TaskError) or is_dependency_error(status)
            for status in self.task_results.values()
        )

    def modality_summary(self) -> list[tuple[str, str, str]]:
        """Summarize the results per modality.

        Returns:
            A list of (status, modality, failures) rows, one per modality.
        """
        task_modalities = {
            task.metadata.name: task.metadata.modalities
            for task in ALL_MOCK_TASK_TEST_GRID
        }

        md_rows = []
        for modality in MODALITIES:
            # tasks which were both compatible with the model and actually ran
            run_tasks = [
                name
                for name, status in self.items()
                if status is not None
                and modality in task_modalities.get(name, [])
                and not (isinstance(status, TaskError) and is_dependency_error(status))
            ]

            if not run_tasks:
                md_rows.append(("skipped", modality, ""))
                continue

            failed = [
                name for name in run_tasks if not isinstance(self[name], TaskResult)
            ]
            passed_count = len(run_tasks) - len(failed)
            total_count = len(run_tasks)

            if not failed:
                md_rows.append((f"✓ ({passed_count}/{total_count})", modality, ""))
            else:
                md_rows.append(
                    (f"✗ ({passed_count}/{total_count})", modality, ", ".join(failed))
                )
        return md_rows

    def to_markdown(self) -> str:
        """Convert the results into a markdown report.

        Returns:
            The markdown report as a string.
        """
        task_modalities = {
            task.metadata.name: task.metadata.modalities
            for task in ALL_MOCK_TASK_TEST_GRID
        }

        md_lines = [
            f"# MTEB Mock-Run Results for `{self.model_name}`",
            "",
            "| Task | Modality | Pass | Reason |",
            "| --- | --- | --- | --- |",
        ]
        for name, status in self.items():
            if status is None:  # not compatible with the model
                continue
            mods = ", ".join(task_modalities.get(name, []))
            if isinstance(status, TaskResult):
                status_str, reason = "✓", "-"
            elif is_dependency_error(status):
                status_str = "skipped"
                reason = status.exception.replace("\n", " ")
            else:
                status_str = "✗"
                reason = status.exception.replace("\n", " ")
            md_lines.append(f"| {name} | {mods} | {status_str} | {reason} |")

        md_rows = self.modality_summary()
        max_status_len = max([len(row[0]) for row in md_rows] + [len("Pass")])
        max_mod_len = max([len(row[1]) for row in md_rows] + [len("Modality")])

        md_lines += [
            "",
            "## Summary by Modality",
            "",
            f"| {'Pass':<{max_status_len}} | {'Modality':<{max_mod_len}} | Failures |",
            f"| {'-' * max_status_len} | {'-' * max_mod_len} | {'-' * 8} |",
        ]
        for status_str, modality, failures in md_rows:
            md_lines.append(
                f"| {status_str:<{max_status_len}} | {modality:<{max_mod_len}} | {failures} |"
            )

        return "\n".join(md_lines)


def get_compatible_mock_tasks(
    model: EncoderProtocol | CrossEncoderProtocol | SearchProtocol,
) -> list[AbsTask]:
    """Get the mock tasks that are compatible with the modalities and protocols of a model.

    Args:
        model: The model object to check.

    Returns:
        The subset of the mock task grid that the model can be evaluated on.
    """

    compatible_tasks = []
    for task in ALL_MOCK_TASK_TEST_GRID:
        if not all(
            mod in model.mteb_model_meta.modalities for mod in task.metadata.modalities
        ):
            continue

        # SearchProtocol / Retrieval-only models (like BM25)
        if isinstance(model, SearchProtocol) and not isinstance(model, EncoderProtocol):
            if not task._support_search:
                continue

        # CrossEncoder models
        if isinstance(model, CrossEncoderProtocol) and not task._support_cross_encoder:
            continue

        compatible_tasks.append(task)
    return compatible_tasks


def mock_run(
    model: EncoderProtocol | CrossEncoderProtocol | SearchProtocol,
) -> MockRunResults:
    """Check a model implementation using mock tasks.

    The mock tasks run locally and do not download any datasets, making this a fast way of
    validating that a model implementation works as intended.

    Args:
        model: The model object to check.

    Returns:
        The [`MockRunResults`][mteb.mocks.mock_run.MockRunResults] of the run, mapping every mock
        task name to its status; a `TaskResult` (the task ran successfully), a `TaskError` (the
        task failed) or `None` (the task is not compatible with the model).
    """
    model_name = model.mteb_model_meta.name or "unknown model"
    compatible_tasks = get_compatible_mock_tasks(model)
    task_status: dict[str, TaskResult | TaskError | None] = {
        task.metadata.name: None for task in ALL_MOCK_TASK_TEST_GRID
    }

    if not compatible_tasks:
        logger.warning(
            "No compatible mock tasks found for model modalities and protocols."
        )
        return MockRunResults(model_name=model_name, task_results=task_status)

    results = evaluate(
        model,
        compatible_tasks,
        cache=None,
        raise_error=False,
        show_progress_bar=False,
    )

    passed_tasks = {res.task_name: res for res in results.task_results}
    failed_tasks = {exc.task_name: exc for exc in (results.exceptions or [])}

    for task in compatible_tasks:
        name = task.metadata.name
        if name in passed_tasks:
            task_status[name] = passed_tasks[name]
        elif name in failed_tasks:
            task_status[name] = failed_tasks[name]

    return MockRunResults(model_name=model_name, task_results=task_status)


def is_dependency_error(error: TaskError) -> bool:
    """Check whether a task failed because an optional dependency was not installed."""
    return any(pat in error.exception.lower() for pat in _DEPENDENCY_ERROR_PATTERNS)
