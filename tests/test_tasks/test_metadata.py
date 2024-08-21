from __future__ import annotations

import pytest

from tests.test_benchmark.task_grid import MOCK_TASK_TEST_GRID


@pytest.mark.parametrize("task", MOCK_TASK_TEST_GRID)
def test_descriptive_stats(task):
    result_stat = task.calculate_metadata_metrics()
    task_stat = task.metadata.descriptive_stats
    for key, value in result_stat.items():
        assert key in task_stat
        assert value == task_stat[key]
