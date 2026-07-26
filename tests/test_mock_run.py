"""Tests for the mock task based model implementation checks (`mteb.check_model_implementation`)."""

from __future__ import annotations

from mteb.mocks import MOCK_MIEB_TASK_GRID, MOCK_TASK_TEST_GRID
from mteb.mocks.mock_run import (
    all_checks_passed,
    get_modality_summary,
    is_dependency_error,
)
from mteb.results.task_result import TaskError, TaskResult

TEXT_TASK = MOCK_TASK_TEST_GRID[0].metadata.name
OTHER_TEXT_TASK = MOCK_TASK_TEST_GRID[1].metadata.name
IMAGE_TASK = MOCK_MIEB_TASK_GRID[0].metadata.name


def _task_result(task_name: str) -> TaskResult:
    return TaskResult(
        dataset_revision="rev",
        task_name=task_name,
        mteb_version="1.0.0",
        scores={},
        evaluation_time=0.0,
    )


def test_all_checks_passed_ignores_dependency_errors():
    results = {
        TEXT_TASK: _task_result(TEXT_TASK),
        OTHER_TEXT_TASK: TaskError(
            task_name=OTHER_TEXT_TASK,
            exception="ImportError: please install 'librosa'",
        ),
        IMAGE_TASK: None,
    }

    assert is_dependency_error(results[OTHER_TEXT_TASK])
    assert all_checks_passed(results)

    results[OTHER_TEXT_TASK] = TaskError(
        task_name=OTHER_TEXT_TASK, exception="ValueError: something went wrong"
    )
    assert not is_dependency_error(results[OTHER_TEXT_TASK])
    assert not all_checks_passed(results)


def test_get_modality_summary():
    results = {
        TEXT_TASK: _task_result(TEXT_TASK),
        OTHER_TEXT_TASK: TaskError(
            task_name=OTHER_TEXT_TASK, exception="ValueError: something went wrong"
        ),
        IMAGE_TASK: None,  # not compatible with the model
    }

    summary = {
        modality: (status, failures)
        for status, modality, failures in get_modality_summary(results)
    }

    assert summary["text"] == ("✗ (1/2)", OTHER_TEXT_TASK)
    # modalities without any compatible task are reported as skipped
    for modality in ["image", "audio", "video"]:
        assert summary[modality] == ("skipped", "")
