from __future__ import annotations

try:
    from tests.task_grid import (
        ALL_TASK_TEST_GRID as all_test_tasks,
    )
    from tests.task_grid import (
        MOCK_MAEB_TASK_GRID as test_tasks_audio,
    )
    from tests.task_grid import (
        MOCK_MIEB_TASK_GRID as test_tasks_image,
    )
    from tests.task_grid import (
        MOCK_MVEB_TASK_GRID as test_tasks_video,
    )
    from tests.task_grid import (
        MOCK_TASK_TEST_GRID as test_tasks,
    )
    from tests.task_grid import (
        MOCK_TASK_TEST_GRID as test_tasks_text,
    )
except ImportError as e:
    raise ImportError(
        "mteb.tests is only available for local development and requires the tests folder from the repository."
    ) from e

__all__ = [
    "all_test_tasks",
    "test_tasks",
    "test_tasks_audio",
    "test_tasks_image",
    "test_tasks_text",
    "test_tasks_video",
]
