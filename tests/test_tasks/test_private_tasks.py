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


def test_accepted_private_tasks_exist():
    """Test that all tasks in ACCEPTED_PRIVATE_TASKS actually exist and are private."""
    if not ACCEPTED_PRIVATE_TASKS:
        pytest.skip("No accepted private tasks configured")

    # Get all tasks including private ones
    all_tasks = get_tasks(exclude_private=False)
    task_names = {task.metadata.name for task in all_tasks}

    for accepted_task in ACCEPTED_PRIVATE_TASKS:
        # Check that the task exists
        assert accepted_task in task_names, (
            f"Accepted private task '{accepted_task}' does not exist in the task registry"
        )

        # Check that it's actually private
        task = next(t for t in all_tasks if t.metadata.name == accepted_task)
        assert task.metadata.is_public is False, (
            f"Task '{accepted_task}' is in ACCEPTED_PRIVATE_TASKS but is not private (is_public={task.metadata.is_public})"
        )
