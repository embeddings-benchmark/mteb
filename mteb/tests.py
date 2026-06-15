from __future__ import annotations

from mteb.tests_mock.task_grid import (
    ALL_TASK_TEST_GRID,
    MOCK_MAEB_TASK_GRID,
    MOCK_MIEB_TASK_GRID,
    MOCK_MVEB_TASK_GRID,
    MOCK_TASK_TEST_GRID,
)

all_test_tasks = ALL_TASK_TEST_GRID
test_tasks_audio = MOCK_MAEB_TASK_GRID
test_tasks_image = MOCK_MIEB_TASK_GRID
test_tasks_video = MOCK_MVEB_TASK_GRID
test_tasks = MOCK_TASK_TEST_GRID
test_tasks_text = MOCK_TASK_TEST_GRID

__all__ = [
    "all_test_tasks",
    "test_tasks",
    "test_tasks_audio",
    "test_tasks_image",
    "test_tasks_text",
    "test_tasks_video",
]
