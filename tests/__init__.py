from __future__ import annotations

from .mock_tasks import MockTask
from .task_grid import MOCK_TASK_TEST_GRID

# Expose lists of test tasks for model validation
test_tasks = MOCK_TASK_TEST_GRID

__all__ = [
    "MockTask",
    "test_tasks",
]
