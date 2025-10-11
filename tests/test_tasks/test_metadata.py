import pytest

from tests.test_benchmark.task_grid import ALL_TASK_TEST_GRID


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
