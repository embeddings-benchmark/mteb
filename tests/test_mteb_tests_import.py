import mteb
from mteb.tests.task_grid import MOCK_TASK_TEST_GRID
from tests.mock_models import MockSentenceTransformer


def test_mteb_tests_import():
    assert isinstance(MOCK_TASK_TEST_GRID, list)
    assert len(MOCK_TASK_TEST_GRID) > 0

    model = MockSentenceTransformer()
    task = MOCK_TASK_TEST_GRID[0]
    results = mteb.evaluate(model, task, cache=None)
    assert len(results.task_results) == 1
