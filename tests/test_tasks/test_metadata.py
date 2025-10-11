"""Tests for testing the descriptive statistics calculation of all tasks"""

from __future__ import annotations

import pytest

from tests.integration_tests.task_grid import ALL_TASK_TEST_GRID


@pytest.mark.parametrize(
    "task",
    ALL_TASK_TEST_GRID,
)
def test_descriptive_stats(task):
    result_stat = task.calculate_descriptive_statistics()
    # remove descriptive task file
    task.metadata.descriptive_stat_path.unlink()
    task_stat = task.expected_stats
    print(task.metadata.name)
    print(result_stat)

    for key, value in result_stat.items():
        assert key in task_stat
        assert value == task_stat[key]
