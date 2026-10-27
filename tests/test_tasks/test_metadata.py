"""Test if the metadata of all tasks is filled and valid."""

import pytest

from mteb.abstasks import AbsTask, AbsTaskRetrieval
from mteb.get_tasks import get_tasks
from tests.test_tasks._tasks_with_unclear_license import (
    TASKS_WITH_UNCLEAR_LICENSE,
)

# Historic datasets without filled metadata. Do NOT add new datasets to this list.
# Tasks should be removed from this list once their metadata is filled.
_HISTORIC_DATASETS = []


def _assert_license_policy(task: AbsTask) -> None:
    """Known tasks may keep an unclear license. New tasks may not."""
    name = task.metadata.name
    if name in TASKS_WITH_UNCLEAR_LICENSE:
        assert task.metadata.has_unclear_license(), (
            f"Task {name} is listed as having an unclear license, but its license "
            f"is {task.metadata.license!r}. Remove it from tests/test_tasks/_tasks_with_unclear_license.py."
        )
        return
    task.metadata._validate_license()


# TODO https://github.com/embeddings-benchmark/mteb/issues/4378
@pytest.mark.parametrize(
    "task",
    get_tasks(exclude_superseded=False, exclude_aggregate=False, exclude_beta=False),
    ids=lambda x: x.metadata.name,
)
def test_all_metadata_is_filled_and_valid(task: AbsTask):
    # --- test metadata is filled and valid ---
    if task.metadata.name not in _HISTORIC_DATASETS:
        task.metadata._validate_metadata()
        assert task.metadata.is_filled(), (
            f"Metadata for {task.metadata.name} is not filled"
        )
    else:
        assert not task.metadata.is_filled(), (
            f"Metadata for {task.metadata.name} is stated as not filled (historic), but it is filled, please remove the dataset from the historic list."
        )

    _assert_license_policy(task)

    # --- Check that no dataset trusts remote code ---
    assert task.metadata.dataset.get("trust_remote_code", False) is False, (
        f"Dataset {task.metadata.name} should not trust remote code"
    )

    # --- Check simplifiedtasktypes is valid ---
    assert isinstance(task.metadata.simplified_task_type, str)

    # --- Test is descriptive stats are present for all datasets ---
    if task.is_aggregate:  # aggregate tasks do not have descriptive stats
        assert set(task.metadata.hf_subsets_to_langscripts.keys()) == {"default"}, (
            f"Aggregate task {task.metadata.name} has incorrect eval_langs, only a default subset is allowed"
        )
        return

    assert task.metadata.descriptive_stats is not None, (
        f"Dataset {task.metadata.name} should have descriptive stats. You can add metadata to your task by running `YourTask().calculate_descriptive_statistics()`"
    )
    assert task.metadata.n_samples is not None

    # Retrieval tasks and TERRa.V2 have a dict prompt, but other tasks should not
    if (
        task.metadata.prompt is not None
        and isinstance(task.metadata.prompt, dict)
        and not (
            isinstance(task, AbsTaskRetrieval) or task.metadata.name in ["TERRa.V2"]  # noqa: PLR6201
        )
    ):
        # Retrieval tasks and TERRa.V2 have a dict prompt, but other tasks should not
        raise ValueError(
            f"Task {task.metadata.name} has a dict prompt, but it should be a string. Please check the metadata of the task."
        )


def test_unclear_licenses_limited_to_existing_tasks():
    """New tasks need a concrete license. Tasks already in the list may keep one."""
    tasks = get_tasks(
        exclude_superseded=False,
        exclude_aggregate=False,
        exclude_beta=False,
        exclude_private=False,
    )
    names = set()
    for task in tasks:
        names.add(task.metadata.name)
        _assert_license_policy(task)

    missing = sorted(TASKS_WITH_UNCLEAR_LICENSE - names)
    assert not missing, (
        "These tasks are on the unclear license list but are not in the benchmark. "
        "Remove them from tests/test_tasks/_tasks_with_unclear_license.py: "
        + ", ".join(missing)
    )
