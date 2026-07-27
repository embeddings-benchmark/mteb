"""Tests for the mock task based model implementation checks (`mteb.mock_run`)."""

from __future__ import annotations

from mteb.mocks import MOCK_MIEB_TASK_GRID, MOCK_TASK_TEST_GRID
from mteb.mocks.mock_run import MockRunResults, is_dependency_error
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


def _results(failure: TaskError) -> MockRunResults:
    return MockRunResults(
        model_name="my_model",
        task_results={
            TEXT_TASK: _task_result(TEXT_TASK),
            OTHER_TEXT_TASK: failure,
            IMAGE_TASK: None,  # not compatible with the model
        },
    )


def test_all_passed_ignores_dependency_errors():
    dependency_error = TaskError(
        task_name=OTHER_TEXT_TASK, exception="ImportError: please install 'librosa'"
    )
    assert is_dependency_error(dependency_error)
    assert _results(dependency_error).all_passed

    failure = TaskError(
        task_name=OTHER_TEXT_TASK, exception="ValueError: something went wrong"
    )
    assert not is_dependency_error(failure)
    assert not _results(failure).all_passed


def test_modality_summary():
    results = _results(
        TaskError(task_name=OTHER_TEXT_TASK, exception="ValueError: something wrong")
    )

    summary = {
        modality: (status, failures)
        for status, modality, failures in results.modality_summary()
    }

    assert summary["text"] == ("✗ (1/2)", OTHER_TEXT_TASK)
    # modalities without any compatible task are reported as skipped
    for modality in ["image", "audio", "video"]:
        assert summary[modality] == ("skipped", "")


def test_to_markdown():
    results = _results(
        TaskError(
            task_name=OTHER_TEXT_TASK, exception="ValueError: something\nwent wrong"
        )
    )

    markdown = results.to_markdown()

    assert "# MTEB Mock-Run Results for `my_model`" in markdown
    assert f"| {TEXT_TASK} | text | ✓ | - |" in markdown
    assert (
        f"| {OTHER_TEXT_TASK} | text | ✗ | ValueError: something went wrong |"
        in markdown
    )
    assert IMAGE_TASK not in markdown
    assert "## Summary by Modality" in markdown


def test_results_are_dict_like():
    results = _results(TaskError(task_name=OTHER_TEXT_TASK, exception="ValueError"))

    assert isinstance(results[TEXT_TASK], TaskResult)
    assert results[IMAGE_TASK] is None
    assert TEXT_TASK in results
    assert dict(results.items()) == results.task_results
