from __future__ import annotations

import pytest

from mteb.overview import get_tasks

# List of accepted private tasks - update this list as needed
ACCEPTED_PRIVATE_TASKS = [
    # Add task names here that are allowed to be private
    # Example: "SomePrivateTask"
]


def test_private_tasks_fail_unless_accepted():
    """Test that private tasks (is_public=False) fail unless they are in the accepted list."""
    # Get all tasks including private ones
    all_tasks = get_tasks(exclude_private=False)

    # Find all private tasks
    private_tasks = [task for task in all_tasks if task.metadata.is_public is False]

    # Check that all private tasks are in the accepted list
    for task in private_tasks:
        assert task.metadata.name in ACCEPTED_PRIVATE_TASKS, (
            f"Private task '{task.metadata.name}' is not in the accepted private tasks list. "
            f"Either make the task public (is_public=True) or add it to ACCEPTED_PRIVATE_TASKS."
        )


pytest.mark.parametrize("task_name", ACCEPTED_PRIVATE_TASKS)
def test_accepted_private_task_exist(task_name: str):
    """Test that all tasks in ACCEPTED_PRIVATE_TASKS actually exist and are private."""
    task = get_task(task_name)
    assert task.metadata.is_public == (
            f"Task '{accepted_task}' is in ACCEPTED_PRIVATE_TASKS but is not private (is_public={task.metadata.is_public})"
        )
